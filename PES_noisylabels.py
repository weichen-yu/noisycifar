"""
For noisylabels datasets

Code is based on PES_semi.py
1. Change default lambda_u 5 for CIFAR-10 and 75 for CIFAR-100.
2. Change Network to ResNet-34.
3. Keep hyper-pamaters of optimizer and PES.

"""

import os
import os.path
import argparse
import random
from selectors import EpollSelector
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import aug_lib
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import LambdaLR

from networks.ResNet import ResNet34, ModelEMA
from common.tools import AverageMeter, getTime, evaluate, predict_softmax, train, task2_detection
from common.NoisyUtil import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset

augment = aug_lib.TrivialAugment()


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=1, type=float, help='data number percent')
parser.add_argument('--noise_type', default='aggre_label', type=str, help='worse_label, aggre_label, random_label1, random_label2, random_label3')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=5e-4)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--optim', default='cos', type=str, help='step, cos')

parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--PES_lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--T1', default=0, type=int, help='if 0, set in below')
parser.add_argument('--T2', default=5, type=int, help='default 5')

parser.add_argument('--ema', default=0.997, type=float, help='EMA decay rate')

args = parser.parse_args()
print(args)
os.system('nvidia-smi')


args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    # cudnn.benchmark = True


def create_model(num_classes=10):
    model = ResNet34(num_classes)
    model.cuda()
    return model


def write_logs(title, args, best_test_acc):
    f = open("./logs/results.txt", "a")
    if args is not None:
        f.write("\n" + getTime() + " " + str(args) + "\n")
    f.write(getTime() + " " + title + " seed-" + str(args.seed) + ", Best Test Acc: " + str(best_test_acc) + "\n")
    f.close()

    
def linear_rampup(current, warm_up=20, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


def MixMatch_train(epoch, net, ema_model, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights, clean_targets, ceriation):
    net.train()
    if epoch >= args.num_epochs / 2:
        args.alpha = 0.75

    losses = AverageMeter('Loss', ':6.2f')
    losses_lx = AverageMeter('Loss_Lx', ':6.2f')
    losses_lu = AverageMeter('Loss_Lu', ':6.5f')

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000 / args.batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, targets_x, index_x = labeled_train_iter.next()
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, targets_x, index_x = labeled_train_iter.next()

        try:
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, inputs_x2, targets_x = inputs_x.cuda(), inputs_x2.cuda(), targets_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs_u11, outputs2_u11 = net(inputs_u)
            outputs_u12, outputs2_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / args.T)  # temparature sharpening
            pu2 = (torch.softmax(outputs2_u11, dim=1) + torch.softmax(outputs2_u12, dim=1)) / 2
            ptu2 = pu2**(1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()
            targets_u2 = ptu2 / ptu2.sum(dim=1, keepdim=True)  # normalize
            targets_u2 = targets_u2.detach()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        all_targets2 = torch.cat([targets_x, targets_x, targets_u2, targets_u2], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        target_a2, target_b2 = all_targets2, all_targets2[idx]

        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b
        mixed_target2 = mixmatch_l * target_a2 + (1 - mixmatch_l) * target_b2

        logits, logits2 = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]
        logits_x2 = logits2[:batch_size * 2]
        logits_u2 = logits2[batch_size * 2:]

        # Lx1 = ceriation(logits_x, mixed_target[:batch_size * 2], False)
        Lx_mean = -torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0)
        Lx2 = torch.sum(Lx_mean * class_weights)
        Lx = Lx2

        # Lx12 = ceriation(logits_x2, mixed_target2[:batch_size * 2], False)
        Lx2_mean = -torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target2[:batch_size * 2], 0)
        Lx222 = torch.sum(Lx2_mean * class_weights)
        Lx22 = Lx222
        Lx = Lx + Lx22

        probs_u = torch.softmax(logits_u, dim=1)
        Lu1 = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        probs_u2 = torch.softmax(logits_u2, dim=1)
        Lu2 = torch.mean((probs_u2 - mixed_target[batch_size * 2:])**2)
        Lu = Lu1 + Lu2
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter, args.T1) * Lu
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(net)

        losses_lx.update(Lx.item(), batch_size * 2)
        losses_lu.update(Lu.item(), len(logits) - batch_size * 2)
        losses.update(loss.item(), len(logits))

    print(losses, losses_lx, losses_lu)


def splite_confident(outs, clean_targets, noisy_targets):
    probs, preds = torch.max(outs.data, 1)


    confident_correct_num = 0
    unconfident_correct_num = 0
    confident_indexs = []
    unconfident_indexs = []

    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:
            confident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                confident_correct_num += 1
        else:
            unconfident_indexs.append(i)
            if clean_targets[i] == preds[i]:
                unconfident_correct_num += 1
    
    precision = unconfident_correct_num/len(unconfident_indexs)
    recall = unconfident_correct_num/(len(clean_targets) - (clean_targets == noisy_targets).sum())
    F1 = 2/(1/precision+1/recall)
    # print(getTime(), "confident and unconfident num:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2), len(unconfident_indexs), round(unconfident_correct_num / len(unconfident_indexs) * 100, 2))
    # import pdb; pdb.set_trace()
    return confident_indexs, unconfident_indexs, precision, recall, F1


def update_trainloader(model, train_data, clean_targets, noisy_targets):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model)

    confident_indexs, unconfident_indexs, precision, recall, F1 = splite_confident(soft_outs, clean_targets, noisy_targets)
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], transform_train)

    uncon_batch = int(args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * args.batch_size)
    con_batch = args.batch_size - uncon_batch

    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss.
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda()
    # print("Category", train_nums, "precent", class_weights)
    return labeled_trainloader, unlabeled_trainloader, class_weights, precision, recall, F1


def noisy_refine(model, clean_labels, noisy_labels, ema_model, train_loader, num_layer, refine_times, ceriation):
    if refine_times <= 0:
        return model
    # frezon all layers and add a new final layer
    for param in model.parameters():
        param.requires_grad = False

    model.renew_layers(num_layer)
    model.cuda()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.PES_lr)
    for epoch in range(refine_times):
        train(model, ema_model, train_loader, optimizer_adam, ceriation, epoch, args.dataset)
        _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")

    for param in model.parameters():
        param.requires_grad = True

    return model, ema_model


if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':
    if args.T1 == 0:
        args.T1 = 20
    args.num_class = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        augment,
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
    
    args.lambda_u = 5
    # For CIFAR-10N noisy labels
    data = train_set.data
    cifar_n_label = torch.load('data/CIFAR-N/CIFAR-10_human.pt') 
    clean_labels = cifar_n_label['clean_label'] 
    
    if args.noise_type == "worse_label":
        noisy_labels = cifar_n_label['worse_label']
    elif args.noise_type == "aggre_label":
        noisy_labels = cifar_n_label['aggre_label']
    elif args.noise_type == "random_label1":
        noisy_labels = cifar_n_label['random_label1']
    elif args.noise_type == 'random_label2':
        noisy_labels = cifar_n_label['random_label2']
    elif args.noise_type == 'random_label3':
        noisy_labels = cifar_n_label['random_label3']

elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':
    if args.T1 == 0:
        args.T1 = 35
    args.num_class = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        augment,
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)
    
    args.lambda_u = 75
    # For CIFAR-100N noisy labels
    data = train_set.data
    cifar_n_label = torch.load('data/CIFAR-N/CIFAR-100_human.pt') 
    clean_labels = cifar_n_label['clean_label'] 
    noisy_labels = cifar_n_label['noisy_label'] 


def save(model, optimizer, scheduler, dataset, noise_type='c100'):
    os.makedirs('checkpoint', exist_ok=True)
    torch.save(model.state_dict(),
                os.path.join('checkpoint', 
                        '{}-{}-best_ckp.ptm'.format(dataset, noise_type)))
    #torch.save(model.state_dict(),
    #            os.path.join('checkpoint', 
    #                    '{}-{:0>3}-encoder.ptm'.format(save_name, epoch)))
    #torch.save([optimizer.state_dict(), scheduler.state_dict()],
    #            os.path.join('checkpoint', 
    #                    '{}-{:0>5}-optimizer.ptm'.format(save_name, epoch)))

ceriation = nn.CrossEntropyLoss().cuda()
train_dataset = Train_Dataset(data, noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)
#task2_eva_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=8, pin_memory=True, drop_last=True)


model = create_model(num_classes=args.num_class)

if args.ema > 0:
    ema_model = ModelEMA(model, args.ema)
else:
    ema_model = None

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

if args.optim == 'cos':
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
else:
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

best_test_acc = 0
test_acc = 0
ema_test_acc = 0
for epoch in range(args.num_epochs):
    if epoch < args.T1:
        train(model, ema_model, train_loader, optimizer, ceriation, epoch, args.dataset)
    else:
        if epoch == args.T1:
            model, ema_model = noisy_refine(model, clean_labels, noisy_labels, ema_model, train_loader, 0, args.T2, ceriation)

        if ema_test_acc > test_acc:
            labeled_trainloader, unlabeled_trainloader, class_weights, precision, recall, F1 = update_trainloader(ema_model, data, clean_labels, noisy_labels)
            print('task2 precision {},recall {},F1 {}'.format(precision, recall, F1))

        else:
            labeled_trainloader, unlabeled_trainloader, class_weights, precision, recall, F1 = update_trainloader(model, data, clean_labels, noisy_labels)
            print('task2 precision {},recall {},F1 {}'.format(precision, recall, F1))

        MixMatch_train(epoch, model, ema_model, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights, clean_labels, ceriation)

    _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
    _, ema_test_acc = evaluate(ema_model, test_loader, ceriation, "Epoch " + str(epoch) + "  EMA Test Acc:")
    # args.T2=0
   
    if best_test_acc < test_acc:
       best_test_acc = test_acc
       save(model, optimizer, scheduler, args.dataset, args.noise_type)
    else:
       best_test_acc = best_test_acc

    if best_test_acc < ema_test_acc:
       best_test_acc = ema_test_acc
       save(ema_model, optimizer, scheduler, args.dataset, args.noise_type)
    else:
       best_test_acc = best_test_acc

    scheduler.step()

print(getTime(), "Best Test Acc:", best_test_acc)

write_logs("Noisylabels:", args, best_test_acc)
