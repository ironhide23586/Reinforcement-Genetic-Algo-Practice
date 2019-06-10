import os
from glob import glob
import pickle
import fnmatch

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from numpy_neural.generator import TrainFeeder
from numpy_neural.shm_nn import FullyConnectedNeuralNet, to_one_hot

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
BATCH_SIZE = 32
NUM_EPOCHS = 1000
EVAL_FREQ_ITERS = 50
MODEL_SAVE_NAME = 'cifar10_fullyconnected.nn'

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


def make_data_nn_consumable(X, Y):
    x = X / 255.
    batch_size = x.shape[0]
    x = x.reshape([batch_size, -1])
    y = [class_name_id_mappings[n] for n in Y]
    y = to_one_hot(np.array(y), len(CLASS_NAMES))
    return x, y


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

    train_fpaths = glob(train_dir + os.sep + '*')
    train_data_reader = TrainFeeder(train_fpaths, batch_size=BATCH_SIZE)

    val_fpaths = glob(val_dir + os.sep + '*')
    val_data_reader = TrainFeeder(val_fpaths, batch_size=1000, shuffle=False, batches_per_queue=1)
    val_x, val_y = val_data_reader.dequeue()
    xv, yv = make_data_nn_consumable(val_x, val_y)
    yv_truth = yv.argmax(axis=1)

    if os.path.isfile(MODEL_SAVE_NAME):
        nn = FullyConnectedNeuralNet(load_path=MODEL_SAVE_NAME)
    else:
        nn = FullyConnectedNeuralNet([32 * 32 * 3, 512, 16, 10])
    acc = 0.

    while train_data_reader.train_state['epoch'] <= NUM_EPOCHS:
        train_x, train_y = train_data_reader.dequeue()
        x, y = make_data_nn_consumable(train_x, train_y)
        loss = nn.train_step(x, y)
        print('Loss =', loss, 'Epoch =', train_data_reader.train_state['epoch'],
              ', Batch =', train_data_reader.train_state['batch'],
              ', Total #Iters =', train_data_reader.train_state['total_iters'])
        if train_data_reader.train_state['total_iters'] % EVAL_FREQ_ITERS == 0:
            print('Evaluating...')
            yv_pred = nn.feed_forward(xv).argmax(axis=1)
            new_acc = accuracy_score(yv_truth, yv_pred)
            print('Validation accuracy =', new_acc)
            if new_acc > acc:
                print('Better model obtained, saving model...')
                nn.save(MODEL_SAVE_NAME)
                acc = new_acc
            else:
                print('Not a better model, not saving this one...')
            print('Best accuracy =', acc)