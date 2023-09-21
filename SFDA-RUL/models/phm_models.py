from turtle import forward
from torch.autograd import Function
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torchvision import models


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


""" LSTM Model """
class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),   
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),  
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
            nn.ReLU())

        # for name, param in self.encoder.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.xavier_normal_(param)
        #     else:
        #         nn.init.zeros_(param)
                
    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
        # encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions, features
# model=LSTM_RUL(14, 32, 5, 0.5, True, device)


""" CNN Model """
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class CNN_RUL(nn.Module):
    def __init__(self):
        super(CNN_RUL, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(16, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(256, 256, kernel_size=2, stride=1),
            nn.AdaptiveMaxPool1d(output_size=36),
            Flatten(),
            nn.Dropout(p=0.1)) 
        self.regressor= nn.Sequential(
            nn.Linear(9216, 15),   
            nn.Linear(15, 12), 
            nn.Linear(12, 10), 
            nn.Linear(10, 8),
            nn.Linear(8, 6), 
            nn.Linear(6, 4),   
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid())    
    def forward(self, src):
        features = self.feature_layers(src)
        predictions = self.regressor(features)
        return predictions, features  


"""Resnet"""
class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=False)
        # self.conv1 = model_resnet.conv1
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.dropout = nn.Dropout(0.5)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
                self.regressor = nn.Sequential(self.bottleneck, self.fc, nn.Sigmoid())
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
                self.regressor = nn.Sequential(self.fc, nn.Sigmoid())
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
            self.regressor = nn.Sequential(self.fc, nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.regressor(x)
        return y, x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
            else:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list