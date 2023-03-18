import csv
import datetime
import os
import glob
import pathlib
from tqdm import tqdm
import numpy as np
import torch

def temporal_ordering(frame_combination):
    print(f'This is the combination function')
    print(frame_combination.shape, type(frame_combination))
    N, C, H, W = frame_combination.shape
    frame = torch.empty([C, N, H, W])
    for i in range(C):
        for j in range(N):
            frame[i, j, :, :] = torch.clone(frame_combination[j, i, :, :])


    list_temp = []
    for i in range(frame.shape[0]):
        list_temp.append(frame[i])

    

    return frame

def temporal_reordering():
    pass