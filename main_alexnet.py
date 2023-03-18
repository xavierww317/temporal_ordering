import csv
import datetime
import os
import glob
import pathlib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from time import *
from PIL import Image
import argparse

# self package
import function
import dataloader
import net_alexnet
import quant_and_coding

parser = argparse.ArgumentParser(description='Video_Feature_compression')
parser.add_argument('--gpuid', default=0, type=int, help='gpu id')
parser.add_argument('--batchsize', default=50, type=int, help='batchsize')
parser.add_argument('--qp', default=0, type=int, help='qp')
args = parser.parse_args()

if __name__ == '__main__':
    f = open('./result/alexnet/{}.csv'.format(args.batchsize), 'w', encoding='utf-8')
    f_writer = csv.writer(f)
    qp_list = [0, 12, 22, 32, 42, 51]

    for qp in qp_list:
        device = torch.device('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')

        root = './picture'
        data_loader = dataloader.loaddata(root, 1)
        print(f'Data load is over')

        model = net_alexnet.alexnet()
        model_weight_path = './alexnet.pth'
        model.load_state_dict(torch.load(model_weight_path))
        model.to(device)
        print(f'Model Ready')

        print(model)

        count = 0
        sum_mse = 0
        sum_ori_bits = 0
        sum_compressed_bits = 0
        frame_list = []

        model.eval()
        with torch.no_grad():
            for input, _ in iter(data_loader):
                print(f'The current count is {count}')
                count += 1

                input = input.to(device)
                output = model(input)
                # print(f'The output shape is {output.shape}')

                frame_list.append(output)

                if len(frame_list) == args.batchsize:
                    # transform
                    frame_combination = torch.cat(frame_list,dim=0).to(device)
                    frame_combination_new = function.temporal_ordering(frame_combination).to(device)
                    N, C, H, W = frame_combination_new.shape
                    print(frame_combination_new.shape)
                    break

                    # hevc coding




    #                 for i in range(N):
    #                     temp = torch.clone(frame_combination_new[i]).unsqueeze(0).to(device)
    #                     _ , compressed_bits, ori_bits, mse = quant_and_coding.HEVC_encoding(temp.cpu().detach().numpy(), 8, qp)
    #                     sum_compressed_bits += compressed_bits
    #                     sum_ori_bits += ori_bits
    #                     sum_mse += mse
    #
    #                 frame_list = []
    #             print('\n')
    #     print(f'ori_bits is {sum_ori_bits}')
    #     print(f'compressed_bits is {sum_compressed_bits}')
    #     print(f'mse is {sum_mse}')
    #     f_writer.writerow([qp,
    #                        sum_ori_bits / (N*len(data_loader)),
    #                        sum_compressed_bits / (N*len(data_loader)),
    #                        sum_mse / (N*len(data_loader)),
    #                        sum_compressed_bits / sum_ori_bits
    #                        ])
    # f.close()


