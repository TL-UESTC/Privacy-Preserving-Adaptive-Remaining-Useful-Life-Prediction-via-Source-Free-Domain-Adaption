from cProfile import label
import os.path as osp
import sys, time
from turtle import shape
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft, cwt
sys.path.append("/home/room/WKK/SFDA-RUL/")
from data.dataset import DataSet_phm
import core


def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("/home/room/WKK/SFDA-RUL/data/Wx.pdf")
    
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("/home/room/WKK/SFDA-RUL/data/Tx.pdf")


def preprocess_4_ResNet(select, train_bearings, test_bearings, data_name, gpu):
    dataset = DataSet_phm.load_dataset(name=data_name)
    begin_1 = True
    begin_2 = True
    
    if select == 'train':
        temp_data = dataset.get_value('data', condition={'bearing_name': train_bearings})
        temp_label = dataset.get_value('RUL', condition={'bearing_name': train_bearings})
    elif select == 'test':
        temp_data = dataset.get_value('data', condition={'bearing_name': test_bearings})
        temp_label = dataset.get_value('RUL', condition={'bearing_name': test_bearings})
    else:
        raise ValueError('wrong selection!')

    for i, x in enumerate(temp_label):
        temp_label[i] = np.arange(temp_data[i].shape[0]) + x
        temp_label[i] = temp_label[i][:, np.newaxis, np.newaxis]
        temp_label[i] = temp_label[i] / np.max(temp_label[i])
        # temp_label[i] = temp_label[i][::8]  # when chang 10
        if begin_1 == True:
            temp_y = temp_label[i]
            begin_1 = False
        else:
            temp_y = np.concatenate((temp_y, temp_label[i]), axis=0)
            
    for i, x in enumerate(temp_data):
        temp_data[i] = x[::-1, ]
        temp_data[i] = np.array(temp_data[i])
        temp_data[i] = np.reshape(temp_data[i], (-1, 2))
        
        if begin_2 == True:
            temp_x = temp_data[i]
            begin_2 = False
        else:
            temp_x = np.concatenate((temp_x, temp_data[i]), axis=0)

    temp_y = np.array(temp_y).squeeze()
    temp_x = np.reshape(temp_x, (-1, 2560, 2))
    
    xo = temp_x[0,:,0]
    
    Twxo, Wxo, *_ = ssq_cwt(xo)
    
    plt.plot(xo)
    # plt.axis('on')
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig("/home/room/WKK/SFDA-RUL/data/raw_x.pdf")
    
    viz(xo, Twxo, Wxo)

if __name__ == "__main__":
    ids = ['OC1']
    data_name = 'phm_data3'
    preprocess_4_ResNet('train', 'Bearing1_1', 'Bearing1_1', data_name, False)