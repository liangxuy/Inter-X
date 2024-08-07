import os
import pickle
import h5py
import numpy as np

SRC_H5 = '/data/xuliang/Inter-X/inter-x.hh5'
DEST_H5 = '/data/xuliang/Inter-X/Inter-X_Dataset/inter-x_regen.h5'
label_file = '/data/xuliang/Inter-X/Inter-X_Dataset/annots/interaction_order.pkl'

with open(label_file, 'rb') as handle:
    order_dict = pickle.load(handle)

f_out = h5py.File(DEST_H5, 'w')

with h5py.File(SRC_H5, 'r') as f:
    sample_name = list(f.keys())
    for i, filename in enumerate(sample_name):
        label = order_dict[filename]
        if label == 1:
            tmp = f[filename]
        elif label == 0:
            tmp = np.zeros_like(f[filename])
            tmp[:,:,0:3] = f[filename][:,:,3:6]
            tmp[:,:,3:6] = f[filename][:,:,0:3]
        f_out.create_dataset(filename, data=tmp, dtype='f')
f_out.close()
