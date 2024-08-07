import os
import argparse
import itertools
import h5py
import numpy as np
from tqdm import tqdm

train_file = '/data/xuliang/Inter-X/Inter-X_Dataset/splits/train.txt'
val_file = '/data/xuliang/Inter-X/Inter-X_Dataset/splits/val.txt'
test_file = '/data/xuliang/Inter-X/Inter-X_Dataset/splits/test.txt'

hhi_file = '/data/xuliang/Inter-X/inter-x.h5'

train_list = [line.rstrip('\n') for line in open(train_file, "r").readlines()]
val_list = [line.rstrip('\n') for line in open(val_file, "r").readlines()]
test_list = [line.rstrip('\n') for line in open(test_file, "r").readlines()]

fw_train = h5py.File('/data/xuliang/Inter-X/train.h5', 'w')
fw_val = h5py.File('/data/xuliang/Inter-X/val.h5', 'w')
fw_test = h5py.File('/data/xuliang/Inter-X/test.h5', 'w')

with h5py.File(hhi_file, 'r') as f:
    keys = list(f.keys())
    pbar = tqdm(keys)
    for k in pbar:
        data = f[k][:]
        if k in train_list:
            fw_train.create_dataset(k, data=data, dtype='f4')
        elif k in val_list:
            fw_val.create_dataset(k, data=data, dtype='f4')
        elif k in test_list:
            fw_test.create_dataset(k, data=data, dtype='f4')