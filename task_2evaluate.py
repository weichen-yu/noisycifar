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
import torch.utils.data as tordata
import aug_lib
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

from networks.ResNet import ResNet34, ModelEMA
from common.tools import AverageMeter, getTime, evaluate, predict_softmax, train
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
parser.add_argument('--save_path', default='/noisycifar/checkpoint/CIFAR10-aggre_label-best_ckp.ptm', type=str, help='the checkpoint before')

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
    import pdb;pdb.set_trace()
    precision = unconfident_correct_num/len(unconfident_indexs)
    recall = unconfident_correct_num/(len(clean_targets) - (clean_targets == noisy_targets).sum())
    F1 = 2/(1/precision+1/recall)
    # print(getTime(), "confident and unconfident num:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2), len(unconfident_indexs), round(unconfident_correct_num / len(unconfident_indexs) * 100, 2))
    # import pdb; pdb.set_trace()
    return confident_indexs, unconfident_indexs, precision, recall, F1

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

def task2_detection(model, train_loader, clean_targets):
    #
    model.eval()
    with torch.no_grad():
        logits_all = []
        labels_all = []
        for i, (data) in enumerate(train_loader):
            #
            images = Variable(data[0]).cuda()
            labels = Variable(data[1]).cuda()
            #indexes = Variable(data[2]).cuda()
            

            logist, _ = model(images)
            logits_all.append(logist)
            labels_all.append(labels)
        logits_all = torch.cat(logits_all,dim=0)
        labels_all = torch.cat(labels_all,dim=0)

        #import pdb;pdb.set_trace()
        clean_targets = torch.tensor(clean_targets).cuda()
        
        confident_indexs, unconfident_indexs, precision, recall, F1 = splite_confident(logits_all,clean_targets, labels_all)
        print('task2 precision {},recall {},F1 {}'.format(precision, recall, F1))

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
def load(model, save_path):
    encoder_ckp = torch.load(save_path)
    model.load_state_dict(encoder_ckp)
    return model
ceriation = nn.CrossEntropyLoss().cuda()
train_dataset = Train_Dataset(data, noisy_labels, transform_train)
#train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
#test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)
task2_eva_loader = DataLoader(dataset=train_dataset, batch_size=100,sampler=tordata.sampler.SequentialSampler(train_dataset), num_workers=8, pin_memory=True, drop_last=True)


model = create_model(num_classes=args.num_class)

#save_model = '/home2/dmw/workspace/noisycifar/checkpoint/CIFAR10-aggre_label-best_ckp.ptm'
model = load(model,args.save_path)
task2_detection(model, task2_eva_loader, clean_labels)

