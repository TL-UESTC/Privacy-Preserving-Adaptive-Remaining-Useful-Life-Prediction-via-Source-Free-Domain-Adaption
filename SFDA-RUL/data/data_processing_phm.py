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


def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1'], ['Bearing1_3']
    elif id == "OC2":
        return ['Bearing2_1'], ['Bearing2_6']
    else:
        return ['Bearing3_1'], ['Bearing3_3']
    

def get_src_ids(id):
    if id == "OC1":
        return ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'], ['Bearing3_3']
    

def get_tgt_ids(id):
    if id == "OC1":
        return ['Bearing1_1','Bearing1_2'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1','Bearing2_2'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2'], ['Bearing3_3']    
    

np.set_printoptions(suppress=True)


def ssq_cwt2(x):
    new_x = np.zeros((len(x), 2560, 490))
    for i in range(len(x)):
        Twx0, *_ = ssq_cwt(x[i, :, 0])
        Twx1, *_ = ssq_cwt(x[i, :, 1])
        Twx0, Twx1 = abs(Twx0), abs(Twx1)
        new_x[i, :, :] = np.concatenate((Twx0, Twx1), axis=0).transpose(1, 0)
        print("\rSSQ_CWT:%.2f %%" % (i * 100 / len(x)), end="")
        # x[i, :, :] = x[i, :, :].transpose(0, 2, 1)
    return new_x


def ssq_cwt_res(x):
    new_x = np.zeros((len(x), 2, 224, 224))
    new_x = torch.tensor(new_x)
    for i in range(len(x)):
        Twx0, *_ = ssq_cwt(x[i, :, 0])
        Twx1, *_ = ssq_cwt(x[i, :, 1])
        Twx0, Twx1 = abs(Twx0), abs(Twx1)
        Twx0_resized, Twx1_resized = core.imresize(torch.tensor(Twx0), sizes=(224, 224)), core.imresize(torch.tensor(Twx1), sizes=(224, 224))
        
        x0, x1 = Twx0_resized.unsqueeze(0), Twx1_resized.unsqueeze(0)
        new_x[i, :, :, :] = torch.cat((x0, x1), axis=0)
        print("\rSSQ_CWT:%.2f %%" % (i * 100 / len(x)), end="")
        # x[i, :, :] = x[i, :, :].transpose(0, 2, 1)
    return new_x


def ssq_cwt_gpu(x):
    new_x = np.zeros((len(x), 2, 224, 224))
    new_x = torch.tensor(new_x).cuda()
    x = torch.tensor(x).cuda()
    for i in range(len(x)):
        Twx0, *_ = ssq_cwt(x[i, :, 0])
        Twx1, *_ = ssq_cwt(x[i, :, 1])
        Twx0, Twx1 = abs(Twx0), abs(Twx1)
        Twx0_resized, Twx1_resized = core.imresize(torch.tensor(Twx0), sizes=(224, 224)), core.imresize(torch.tensor(Twx1), sizes=(224, 224))
        
        x0, x1 = Twx0_resized.unsqueeze(0), Twx1_resized.unsqueeze(0)
        new_x[i, :, :, :] = torch.cat((x0, x1), axis=0)
        print("\rSSQ_CWT:%.2f %%" % (i * 100 / len(x)), end="")
        # x[i, :, :] = x[i, :, :].transpose(0, 2, 1)
    return new_x


def minmaxscalar(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)


def standardscalar(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def linearscalar(data):
    max = np.max(data)
    return data / max


class MyDataset(Dataset):
    def __init__(self, pth, pth2):
        self.pth = pth
        self.pth2 = pth2
        self.x_lists = os.listdir(self.pth)
        self.x_lists.sort(key=lambda x:int(x[4:-4]))
        self.y_data = np.loadtxt(self.pth2 + "labels.txt", dtype=int)
        # self.y_data = minmaxscalar(self.y_data)
        
    def __getitem__(self, index):
        # print(self.pth + self.x_lists[index])
        self.x1 = pd.read_csv(self.pth + self.x_lists[index],
                              header=None, usecols=[4]).values
        self.x2 = pd.read_csv(self.pth + self.x_lists[index],
                              header=None, usecols=[5]).values
        # self.x1 = minmaxscalar(self.x1)
        # self.x2 = minmaxscalar(self.x2)
        
        self.x = torch.tensor(np.concatenate((self.x1, self.x2), axis=1))

        return self.x, self.y_data[index]
    
    def __len__(self):
        return len(self.x_lists)
    
    
class MyDataset_NL(Dataset):
    def __init__(self, pth, pth2):
        self.pth = pth
        self.pth2 = pth2
        self.x_lists = os.listdir(self.pth)
        self.x_lists.sort(key=lambda x:int(x[4:-4]))
        self.y_data = np.loadtxt(self.pth2 + "labels.csv")
        
    def __getitem__(self, index):
        # print(self.pth + self.x_lists[index])
        self.x1 = pd.read_csv(self.pth + self.x_lists[index],
                              header=None, usecols=[0]).values
        self.x2 = pd.read_csv(self.pth + self.x_lists[index],
                              header=None, usecols=[1]).values
        
        # self.x1 = self.x1[::10]
        # self.x2 = self.x2[::10]
        
        self.x = torch.tensor(np.concatenate((self.x1, self.x2), axis=1))

        return self.x, self.y_data[index]
    
    def __len__(self):
        return len(self.x_lists)


class MyDataset_new(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __getitem__(self, index: int):
        
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)


def preprocess(select, train_bearings, test_bearings, norm_id, data_name):
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
        
        if norm_id == "minmax":
            minmax = MinMaxScaler()
            temp_data[i] = minmax.fit_transform(temp_data[i])
            # temp_data[i][:,0] = minmaxscalar(temp_data[i][:,0])
            # temp_data[i][:,1] = minmaxscalar(temp_data[i][:,1])
        elif norm_id == "standard":
            temp_data[i][:,0] = standardscalar(temp_data[i][:,0])
            temp_data[i][:,1] = standardscalar(temp_data[i][:,1])
        elif norm_id == "linear":
            temp_data[i][:,0] = linearscalar(temp_data[i][:,0])
            temp_data[i][:,1] = linearscalar(temp_data[i][:,1])
        else:
            pass
        
        if begin_2 == True:
            temp_x = temp_data[i]
            begin_2 = False
        else:
            temp_x = np.concatenate((temp_x, temp_data[i]), axis=0)

    temp_y = np.array(temp_y).squeeze()
    temp_x = np.reshape(temp_x, (-1, 2560, 2))
    if norm_id == "ssq_cwt":
        temp_x = ssq_cwt2(temp_x)

    print(temp_x.shape)

    return temp_x, temp_y


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
    
    if gpu == True:
        temp_x = ssq_cwt_gpu(temp_x)
    else:
        temp_x = ssq_cwt_res(temp_x)
        
    print('\n')
    print(temp_x.shape)

    return temp_x, temp_y


def preprocess_4_online(select, train_bearings, test_bearings, data_name, gpu):
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
    
    if gpu == True:
        temp_x = ssq_cwt_gpu(temp_x)
    else:
        temp_x = ssq_cwt_res(temp_x)
        
    print('\n')
    print(temp_x.shape)
    
    temp_x1 = temp_x[:len(temp_x)//2]
    temp_y1 = temp_y[:len(temp_x)//2]
    
    temp_x2 = temp_x[len(temp_x)//2:]
    temp_y2 = temp_y[len(temp_x)//2:]
    
    # index = np.arange(len(temp_x))
    # np.random.shuffle(index)
    # index1 = index[:len(temp_x)//2]
    # index2 = index[len(temp_x)//2:]
    
    # temp_x1, temp_y1, temp_x2, temp_y2 = [], [], [], []
    # for i in index1:
    #     temp_x1.append(temp_x[i])
    #     temp_y1.append(temp_y[i])
    
    # for i in index2:
    #     temp_x2.append(temp_x[i])
    #     temp_y2.append(temp_y[i])

    return temp_x1, temp_y1, temp_x2, temp_y2


if __name__ == "__main__":
    ids = ['OC1']
    data_name = 'phm_data3'
    
    # for id in ids:
    #     print(f'Processing:{id}...')
    #     train, test = get_ids(id)
    #     train_x, train_y = preprocess_4_ResNet('train', train, test, data_name, False)
    #     np.save(f"/media/room/新加卷/WKK/data/Train_{id}_x.npy", train_x)
    #     np.save(f"/media/room/新加卷/WKK/data/Train_{id}_y.npy", train_y)
        
    #     test_x, test_y = preprocess_4_ResNet('test', train, test, data_name, False)

    #     np.save(f"/media/room/新加卷/WKK/data/Test_{id}_x.npy", test_x)
    #     np.save(f"/media/room/新加卷/WKK/data/Test_{id}_y.npy", test_y)
    
    for id in ids:
        print(f'Processing:{id}...')
        train_src, test_src = get_src_ids(id)
        train_tgt, test_tgt = get_tgt_ids(id)
        src_x, src_y = preprocess_4_ResNet('train', train_src, test_src, data_name, False)
        np.save(f"/media/room/新加卷/WKK/data/source_{id}_x.npy", src_x)
        np.save(f"/media/room/新加卷/WKK/data/source_{id}_y.npy", src_y)
        
    #     # tgt_x, tgt_y = preprocess_4_ResNet('train', train_tgt, test_tgt, data_name, False)
    #     # np.save(f"/home/room/WKK/SFDA-RUL/data/target_{id}_x.npy", tgt_x)
    #     # np.save(f"/home/room/WKK/SFDA-RUL/data/target_{id}_y.npy", tgt_y)
        
    #     tgt_x, tgt_y = preprocess_4_ResNet('test', train_tgt, test_tgt, data_name, False)
    #     np.save(f"/media/room/新加卷/WKK/data/test_{id}_x.npy", tgt_x)
    #     np.save(f"/media/room/新加卷/WKK/data/test_{id}_y.npy", tgt_y)