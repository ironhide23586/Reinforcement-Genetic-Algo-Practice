import os
from glob import glob
import pickle
import fnmatch

import numpy as np
import cv2
from tqdm import tqdm


DATA_DIR = './data/cifar10'
CLASS_NAMES = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

class_name_id_mappings = {CLASS_NAMES[i]: i for i in range(len(CLASS_NAMES))}


def arr2im_bgr(arr):
    im = np.zeros([32, 32, 3])
    for i in range(3):
        im[:, :, 2 - i] = arr[i * 1024: (i + 1) * 1024].reshape([32, 32])
    return im.astype(np.uint8)


def process_cifar10_data_chunk(data, dir):
    chunk_size = data[b'data'].shape[0]
    for i in tqdm(range(chunk_size)):
        im = arr2im_bgr(data[b'data'][i])
        fname = data[b'filenames'][i].decode('utf-8')
        label = data[b'labels'][i]
        ext = '.' + fname.split('.')[-1]
        class_name = CLASS_NAMES[label]
        out_fname = fname.replace(ext, '_' + class_name + ext)
        out_fpath = dir + os.sep + out_fname
        cv2.imwrite(out_fpath, im)


if __name__ == '__main__':
    preprocessed_data_dir = DATA_DIR + os.sep + 'preprocessed'
    train_dir = preprocessed_data_dir + os.sep + 'train'
    val_dir = preprocessed_data_dir + os.sep + 'val'
    if not os.path.isdir(preprocessed_data_dir):
        all_fpaths = glob(DATA_DIR + os.sep + '*')
        train_batch_fpaths = fnmatch.filter(all_fpaths, '*data_batch*')
        val_batch_fpath = DATA_DIR + os.sep + 'test_batch'
        os.makedirs(preprocessed_data_dir)
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        for fpath in train_batch_fpaths:
            data = pickle.load(open(fpath, 'rb'), encoding='bytes')
            process_cifar10_data_chunk(data, train_dir)
        data = pickle.load(open(val_batch_fpath, 'rb'), encoding='bytes')
        process_cifar10_data_chunk(data, val_dir)
    k = 0