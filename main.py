import os
import torch
import argparse
from CountNet import CountingModel
from train import train
from eval import eval


def main(args):
    net = CountingModel(3, 32)
    net = net.cuda()
    if args.mode == 'train':
        train(net, args)

    if args.mode == 'test':
        eval(net, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--alpha', type=int, default=5)
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--path', type=str, default=r'G:\papers\TheCellCount(paper)\CellCounting\multiclass_dataset\MoNuSAC')
    parser.add_argument('--model_save_dir', type=str, default=r'G:\papers\TheCellCount(paper)\CellCounting\weights')

    # test
    parser.add_argument('--test_model', type=str, default=r'weights/Best.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--result_dir', type=str, default=r'result_image')
    parser.add_argument('--test_size', type=int, default=512)

    args = parser.parse_args()
    print(args)
    main(args)
