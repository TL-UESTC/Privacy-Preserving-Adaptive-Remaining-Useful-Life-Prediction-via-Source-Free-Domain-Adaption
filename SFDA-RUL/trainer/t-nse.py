import argparse
import os, sys
import os.path as osp
from turtle import color
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random, pdb, math, copy
sys.path.append(".")
from utils import *
from models.my_models import *
from models.phm_models import ResNetFc
from data.data_processing_phm import *
from torch.utils.data import DataLoader
from data.mydataset import create_dataset_full
import matplotlib.pyplot as plt
from trainer.train_eval import evaluate
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
from sklearn.manifold import TSNE

device = torch.device('cuda:2')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# 设置随机数种子
setup_seed(0)


def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1'], ['Bearing1_3']
    elif id == "OC2":
        return ['Bearing2_1'], ['Bearing2_6']
    else:
        return ['Bearing3_1'], ['Bearing3_3']


def get_features(model, data):
    start_test = True
    for i, data in enumerate(data):
        inputs, _ = data
        src = inputs.to(device).to(torch.float32)
        _, fea = model(src)
        if start_test:
            all_fea = fea.float().cpu()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, fea.float().cpu()), 0)
    return all_fea


def t_SNE():
    
    data_path= "data/cmapps_train_test_cross_domain.pt"
    my_dataset = torch.load(data_path)
    src_train_dl, src_test_dl = create_dataset_full(my_dataset['FD004'], batch_size=256)
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset['FD001'], batch_size=256)

    source_model = LSTM_RUL(14, 32, 3, 0.5, True, device).to(device)
    target_model = LSTM_RUL(14, 32, 3, 0.5, True, device).to(device)
    
    model_path1 = f'//home/room/WKK/SFDA-RUL/trained_models/different_layers_model_best/pretrained_LSTM_3_FD004_new.pt'
    #model_path1 = f'trained_models/single_domain/pretrained_LSTM_FD001_new.pt'
    checkpoint1 = torch.load(model_path1)
    source_model.load_state_dict(checkpoint1['state_dict'])
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    #model_path = f'//home/room/WKK/SFDA-RUL/trained_models/different_layers_model_best/pretrained_LSTM_3_FD003_new.pt'
    model_path = '/home/room/WKK/SFDA-RUL/trained_models/cross_domain_models/bi/FD004_FD001/3_layers/first_model_50.pt'
    #model_path = f'trained_models/single_domain/pretrained_LSTM_FD001_new.pt'
    checkpoint2 = torch.load(model_path)
    target_model.load_state_dict(checkpoint2['state_dict'])
    target_model.eval()
    set_requires_grad(target_model, requires_grad=False)
    
    start_test = True
    #source
    with torch.no_grad():

        for inputs, labels in src_test_dl:
        
            src = inputs.to(device)
            labels = labels.to(device)
            
            _, feas = target_model(src)
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    print(all_fea.size(0))
    #target
    with torch.no_grad():
        for inputs, labels in tgt_test_dl:
            
            src = inputs.to(device)
            labels = labels.to(device)
            
            _, feas = target_model(src)
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    print(all_fea.size(0))
        

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(all_fea)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    import matplotlib.pyplot as plt

    # plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.axis('on')
    # plt.xticks([]) 
    # plt.yticks([])
    
    source_number = 100
    # plt.scatter(X_norm[:source_number,0], np.negative(X_norm[:source_number,1]), marker='o',s=12, label = 'source', color='r')
    # plt.scatter(X_norm[source_number:,0], np.negative(X_norm[source_number:,1]), marker='o',s=12, label = 'target', color='g') 
    
    plt.scatter(X_norm[:source_number,0], X_norm[:source_number,1], marker='o',s=12, label='source', color='r')
    plt.scatter(X_norm[source_number:,0], X_norm[source_number:,1], marker='o',s=12, label='target', color='g') 
    plt.title("FD004->FD001, before adaptation")
    
    plt.legend(["Source", "Target"], fontsize=8) 
    
    # plt.scatter(X_s0[:, 0], np.negative(X_s0[:, 1]), marker='^',s=10, c='k')
    # plt.scatter(X_s1[:, 0], np.negative(X_s1[:, 1]), marker='^',s=10, c='g')
    # plt.scatter(X_s2[:, 0], np.negative(X_s2[:, 1]), marker='^',s=10, c='b')
    # plt.scatter(X_s3[:, 0], np.negative(X_s3[:, 1]), marker='^',s=10, c='r')
    # plt.scatter(X_s4[:, 0], np.negative(X_s4[:, 1]), marker='^',s=10, c='y')
    
    # plt.scatter(X_t0[:, 0], np.negative(X_t0[:, 1]), marker='o',s=10, c='k')
    # plt.scatter(X_t1[:, 0], np.negative(X_t1[:, 1]), marker='o',s=10, c='g')
    # plt.scatter(X_t2[:, 0], np.negative(X_t2[:, 1]), marker='o',s=10, c='b')
    # plt.scatter(X_t3[:, 0], np.negative(X_t3[:, 1]), marker='o',s=10, c='r')
    # plt.scatter(X_t4[:, 0], np.negative(X_t4[:, 1]), marker='o',s=10, c='y')
    
    # plt.scatter(X_s0[:, 0], -X_s0[:, 1], marker='^',s=10, c='k')
    # plt.scatter(X_s1[:, 0], -X_s1[:, 1], marker='^',s=10, c='g')
    # plt.scatter(X_s2[:, 0], -X_s2[:, 1], marker='^',s=10, c='b')
    # plt.scatter(X_s3[:, 0], -X_s3[:, 1], marker='^',s=10, c='r')
    # plt.scatter(X_s4[:, 0], -X_s4[:, 1], marker='^',s=10, c='y')
    
    # plt.scatter(X_t0[:, 0], -X_t0[:, 1], marker='o',s=10, c='k')
    # plt.scatter(X_t1[:, 0], -X_t1[:, 1], marker='o',s=10, c='g')
    # plt.scatter(X_t2[:, 0], -X_t2[:, 1], marker='o',s=10, c='b')
    # plt.scatter(X_t3[:, 0], -X_t3[:, 1], marker='o',s=10, c='r')
    # plt.scatter(X_t4[:, 0], -X_t4[:, 1], marker='o',s=10, c='y')
    
    # plt.legend(["label0-S","label1-S","label2-S","label3-S","label4-S","label0-T","label1-T","label2-T","label3-T","label4-T"], fontsize=8)

    plt.savefig("4_1_test2.pdf")


def t_SNE2():
    
    with torch.no_grad():
        src_id = 'OC3'
        tgt_id = 'OC2'
        print(f'From_source:{src_id}--->target:{tgt_id}...')
        
        data_name = 'phm_data3'
        gpu = False
        
        train_bearings_s, test_bearings_s = get_ids(src_id)
        train_bearings_t, test_bearings_t = get_ids(tgt_id)
        
        test_X_s, test_Y_s = preprocess_4_ResNet(select='test', train_bearings=train_bearings_s, test_bearings=test_bearings_s, data_name=data_name, gpu=gpu)
        test_X_t, test_Y_t = preprocess_4_ResNet(select='test', train_bearings=train_bearings_t, test_bearings=test_bearings_t, data_name=data_name, gpu=gpu)
        
        test_set_s = MyDataset_new(test_X_s, test_Y_s)
        test_set_t = MyDataset_new(test_X_t, test_Y_t)
        test_loader_s = DataLoader(dataset=test_set_s,
                                batch_size=64,
                                shuffle=False,
                                num_workers=0,
                                drop_last=False)
        test_loader_t = DataLoader(dataset=test_set_t,
                                batch_size=64,
                                shuffle=False,
                                num_workers=0,
                                drop_last=False)
        
        source_model = ResNetFc(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        first_model = ResNetFc(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        
        load_path_s = f'/home/room/WKK/SFDA-RUL/trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new.pt'
        checkpoint_s = torch.load(load_path_s, map_location='cuda:2')
        source_model.load_state_dict(checkpoint_s['state_dict'])
        
        first_model = torch.load(f'/home/room/WKK/SFDA-RUL/trained_models/DA_phm/resnet/{src_id}_{tgt_id}/first_model.pth')
        first_model = first_model.to(device)
        source_model.eval()
        first_model.eval()

        all_fea_s = get_features(source_model, test_loader_s)
        all_fea_t1 = get_features(source_model, test_loader_t)
        all_fea_t2 = get_features(first_model, test_loader_t)
        
        all_fea = torch.cat((all_fea_s, all_fea_t1))
        all_fea2 = torch.cat((all_fea_s, all_fea_t2))

    print(all_fea_s.size(0))
    print(all_fea_t1.size(0))

    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(all_fea)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    import matplotlib.pyplot as plt

    # plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.axis('on')
    
    plt.scatter(X_norm[:len(all_fea_s),0], X_norm[:len(all_fea_s),1], marker='o',s=12, label='source', color='r')
    plt.scatter(X_norm[len(all_fea_s):,0], X_norm[len(all_fea_s):,1], marker='o',s=12, label='target', color='g') 
    plt.title(f"{src_id}->{tgt_id}, before adaptation")
    plt.legend(["Source", "Target"], fontsize=8) 
    plt.savefig("test.pdf")
    plt.close()
    
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(all_fea2)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    # plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.axis('on')
    
    plt.scatter(X_norm[:len(all_fea_s),0], X_norm[:len(all_fea_s),1], marker='o',s=12, label='source', color='r')
    plt.scatter(X_norm[len(all_fea_s):,0], X_norm[len(all_fea_s):,1], marker='o',s=12, label='target', color='g') 
    plt.title(f"{src_id}->{tgt_id}, after adaptation")
    plt.legend(["Source", "Target"], fontsize=8) 
    plt.savefig("test2.pdf")
    plt.close()
    

if __name__ == "__main__":

    t_SNE()
    # t_SNE2()