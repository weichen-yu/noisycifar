#CUDA_VISIBLE_DEVICES=4 nohup python PES_noisylabels.py --noise_type worse_label --dataset CIFAR10  > c10_worse.log 2>&1 &

#CUDA_VISIBLE_DEVICES=5 nohup python PES_noisylabels.py --noise_type random_label2 --dataset CIFAR10 > c10_rand2.log 2>&1 &

#CUDA_VISIBLE_DEVICES=6 nohup python PES_noisylabels.py --noise_type aggre_label --dataset CIFAR10 > c10_aggre.log 2>&1 &

#CUDA_VISIBLE_DEVICES=7 nohup python PES_noisylabels.py --dataset CIFAR100 --seed 5 > c100_seed5.log 2>&1  &
# CUDA_VISIBLE_DEVICES=6 nohup python PES_noisylabels.py --dataset CIFAR100 --seed 1 > c100.log 2>&1 


#for task_2
CUDA_VISIBLE_DEVICES=6 python task_2evaluate.py --dataset CIFAR100 --seed 5 --save_path /home2/dmw/workspace/noisycifar/CIFAR100-aggre_label-best_ckp.ptm