import sys
sys.path.append("..")
from utils import *
from models.phm_models import *
from data.data_processing_phm import *
from models.models_config import get_model_config
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
import matplotlib.pyplot as plt
from trainer.train_eval_phm import *
import numpy as np
import pandas as pd
import time
import os
device = torch.device('cuda:1')

seed = 0
 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def discrepancy(pred1, pred2):
    return torch.mean(torch.abs(pred1 - pred2))
    # return torch.sqrt(torch.sum((pred1-pred2**2))) / len(pred1)
    # return torch.norm((pred1 - pred2), p=2, dim=0)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter, PREHEAT_STEPS, hyper):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(hyper['lr'], i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(hyper['lr'], i_iter, hyper['epochs'], hyper['power'])
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter, PREHEAT_STEPS, hyper):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(hyper['lr'], i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(hyper['lr'], i_iter, hyper['epochs'], hyper['power'])
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        
        
def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1'], ['Bearing1_3']
    elif id == "OC2":
        return ['Bearing2_1'], ['Bearing2_6']
    else:
        return ['Bearing3_1'], ['Bearing3_3']


def cross_domain_train(params, device, config, model, src_id, tgt_id, norm_id, data_name, network, gpu, online):
    best_rmse, best_mae, best_score = 100, 100, 0

    hyper = params[f'{src_id}_{tgt_id}']

    df = pd.DataFrame(columns=['wight_loss1', 'dis_loss1', 'wight_loss2', 'dis_loss2', 'rmse', 'mae', 'score'])
    df.to_csv(f"SFDA-RUL/results/loss/{src_id}_{tgt_id}_train_loss2_{hyper['lambda_w']}.csv", index=False)

    print(f'From_source:{src_id}--->target:{tgt_id}...')

    if network == "LSTM_RUL":

        save_path = f'./trained_models/DA_phm/{src_id}_{tgt_id}/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        train_bearings_s, test_bearings_s = get_ids(src_id)
        train_bearings_t, test_bearings_t = get_ids(tgt_id)
        train_X_t, train_Y_t = preprocess(select='train', train_bearings=train_bearings_t, test_bearings=test_bearings_t, norm_id=norm_id, data_name=data_name)
        test_X_t, test_Y_t = preprocess(select='test', train_bearings=train_bearings_t, test_bearings=test_bearings_t, norm_id=norm_id, data_name=data_name)
    elif online == True:
        save_path = f'./trained_models/DA_phm/resnet/{src_id}_{tgt_id}/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path) 

        train_bearings_t, test_bearings_t = get_ids(tgt_id)
        train_X_t, train_Y_t, test_X_t, test_Y_t = preprocess_4_online(select='train', train_bearings=train_bearings_t, test_bearings=test_bearings_t, data_name=data_name, gpu=gpu)
        
    else:
        save_path = f'./trained_models/DA_phm/resnet/{src_id}_{tgt_id}/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path) 
        train_bearings_t, test_bearings_t = get_ids(tgt_id)
        train_X_t, train_Y_t = preprocess_4_ResNet(select='train', train_bearings=train_bearings_t, test_bearings=test_bearings_t, data_name=data_name, gpu=gpu)
        test_X_t, test_Y_t = preprocess_4_ResNet(select='test', train_bearings=train_bearings_t, test_bearings=test_bearings_t,data_name=data_name, gpu=gpu)
    
    train_set_t = MyDataset_new(train_X_t, train_Y_t)
    test_set_t = MyDataset_new(test_X_t, test_Y_t)
    
    train_loader_t = DataLoader(dataset=train_set_t,
                              batch_size=hyper['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)
    test_loader_t = DataLoader(dataset=test_set_t,
                              batch_size=64,
                              shuffle=False,
                              num_workers=0,
                              drop_last=False)
    print('Restore source pre_trained model...')
    
    if network == "LSTM_RUL":
        load_path = f'SFDA-RUL/trained_models/pretrained_phm/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path)
        
        load_path2 = f'SFDA-RUL/trained_models/pretrained_phm/pretrained_{src_id}_new2.pt'
        checkpoint2 = torch.load(load_path2)
        
        first_model = model(2, 32, config['layers'], 0.5, True, device).to(device)

        print('=' * 89)
        print(f'The {config["model_name"]} has {count_parameters(first_model):,} trainable parameters')
        print('=' * 89)
        first_model.load_state_dict(checkpoint['state_dict'])
        encoder = first_model.encoder
        regressor1 = first_model.regressor
        
        # initialize target model
        second_model = model(2, 32, config['layers'], 0.5, True, device).to(device)
        second_model.load_state_dict(checkpoint2['state_dict'])

        # second_model.regressor.4 = nn.Linear(16, 1)
        regressor2 = second_model.regressor
    else:
        load_path = f'SFDA-RUL/trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path, map_location=torch.device('cuda:0'))
        
        load_path2 = f'SFDA-RUL/trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new2.pt'
        checkpoint2 = torch.load(load_path2, map_location=torch.device('cuda:0'))
        
        first_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)

        print('=' * 89)
        print(f'The ResNet has {count_parameters(first_model):,} trainable parameters')
        print('=' * 89)
        first_model.load_state_dict(checkpoint['state_dict'])
        encoder = first_model.feature_layers
        regressor1 = first_model.regressor
        

        # initialize target model
        second_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        second_model.load_state_dict(checkpoint2['state_dict'])
        regressor2 = second_model.regressor
    
    encoder.train()
    regressor1.train()
    regressor2.train()
    
    # criterion
    criterion = RMSELoss()
    criterion2 = nn.MSELoss()
    
    # optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=hyper['lr'], betas=(0.5, 0.9), weight_decay=5e-4)
    regressor1_optim = torch.optim.Adam(regressor1.parameters(), lr=hyper['lr'] * 0.5, betas=(0.5, 0.9))
    regressor2_optim = torch.optim.Adam(regressor2.parameters(), lr=hyper['lr'] * 0.5, betas=(0.5, 0.9))

    scheduler1 = StepLR(encoder_optim, step_size=50, gamma=0.5)
    scheduler2 = StepLR(regressor1_optim, step_size=50, gamma=0.5)
    scheduler3 = StepLR(regressor2_optim, step_size=50, gamma=0.5)

    src_only_loss, src_only_mae, src_only_score, pred_labels, true_labels = evaluate(first_model, regressor1, test_loader_t, device)
    fig1 = plt.figure()
    plt.plot(pred_labels, label='pred labels', linewidth=0.5)
    plt.plot(true_labels, label='true labels', linewidth=0.5)
    plt.legend()
    fig1.savefig('training_images/'+src_id+tgt_id+'so.png')
    plt.close(fig1)
    
    W_E = None
    W_R = None
    for w_e in encoder.parameters():
        if W_E is None:
            W_E = w_e.view(-1)  
        else:
            W_E = torch.cat((W_E, w_e.view(-1)), 0)
    

    for w_r in regressor1.parameters():
        if W_R is None:
            W_R = w_r.view(-1)  
        else:
            W_R = torch.cat((W_R, w_r.view(-1)), 0)
            
    W_E = W_E.detach()
    W_R = W_R.detach()

    weight = hyper['lambda_w']

    all_start_time = time.time()
    for epoch in range(1, hyper['epochs'] + 1):
        encoder.train()
        regressor1.train()
        regressor2.train()
        
        len_target = len(train_loader_t)
        num_iter = len_target

        mse_losses = 0
        loss_dis1_a = 0
        loss_weight1_a = 0
        dis_losses = 0
        dis_m_losses = 0
        start_time = time.time()
        for batch_idx in range(num_iter):  # , leave=False): 
            if batch_idx % len_target == 0:
                iter_target = iter(train_loader_t)
                
            target_x, target_y = iter_target.next()
            target_x = target_x.to(device).to(torch.float32)
            
            # Maximize discrepancy
            for _ in range(config['k_disc']):
                set_requires_grad(encoder, requires_grad=False)
                set_requires_grad(regressor1, requires_grad=True)
                set_requires_grad(regressor2, requires_grad=True)
                
                regressor1.zero_grad()
                regressor2.zero_grad()
                # print(torch.cat((source_x, target_x), 0).shape)

                target_pred1, target_features = first_model(target_x)
                target_pred2 = regressor2(target_features)
                
                loss_dis1 = discrepancy(target_pred1, target_pred2)

                W_R_1 = None
                W_R_2 = None
                
                for (w_r_1, w_r_2) in zip(regressor1.parameters(), regressor2.parameters()):
                    if W_R_1 is None and W_R_2 is None:
                        W_R_1 = w_r_1.view(-1)  
                        W_R_2 = w_r_2.view(-1)
                    else:
                        W_R_1 = torch.cat((W_R_1, w_r_1.view(-1)), 0)
                        W_R_2 = torch.cat((W_R_2, w_r_2.view(-1)), 0)
                
                diss1 = W_R_1 - W_R
                diss2 = W_R_2 - W_R
                
                loss_diss1 = torch.norm(diss1, p=2, dim=0)
                loss_diss2 = torch.norm(diss2, p=2, dim=0)
                
                loss = -loss_dis1
                loss_weight1 =  (loss_diss1 + loss_diss2)/2
                loss += (loss_diss1 + loss_diss2) * hyper['lambda_w'] * 2
                loss.backward()
                regressor1_optim.step()
                regressor2_optim.step()
                loss_dis1_a += loss_dis1.item()
                loss_weight1_a += loss_weight1.item()

            # Minimize discrepancy
            for _ in range(config['k_clf']):

                encoder_optim.zero_grad()
                
                set_requires_grad(encoder, requires_grad=True)
                set_requires_grad(regressor1, requires_grad=False)
                set_requires_grad(regressor2, requires_grad=False)
                
                regressor1.zero_grad()
                regressor2.zero_grad()
                target_pred1, target_features = first_model(target_x)
                target_pred2 = regressor2(target_features)
                
                loss_dis = discrepancy(target_pred1, target_pred2)
                
                W_E_t = None
                for w_e_t in encoder.parameters():
                    if W_E_t is None:
                        W_E_t = w_e_t.view(-1)  
                    else:
                        W_E_t = torch.cat((W_E_t, w_e_t.view(-1)), 0)
                
                diss = W_E_t - W_E
                loss_diss_m = torch.norm(diss, p=2, dim=0)
                
                #total loss
                loss = loss_dis          
                loss = loss + loss_diss_m * hyper['lambda_w']
                dis_m_losses += loss_diss_m.item()
                dis_losses += loss_dis.item()
                loss.backward()
                encoder_optim.step()
        loss_dis1_m = loss_dis1_a / (num_iter * config['k_disc'])
        loss_weight1_m  = loss_weight1_a / (num_iter * config['k_disc'])      
        mean_mse_loss = dis_m_losses / (num_iter * config['k_clf'])
        mean_dis_loss = dis_losses / (num_iter * config['k_clf'])
        # tensorboard logging
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch :02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Weight_loss:{mean_mse_loss} \t dis_loss:{mean_dis_loss}')

        if epoch % 1 == 0:
            save_name1 = f'/first_model_{epoch}.pt'
            save_name2 = f'/second_model_{epoch}.pt'
            test_loss, test_mae, test_score, pred_labels_DA, true_labels = evaluate(first_model, regressor2, test_loader_t,device)
            if test_loss < best_rmse:
                best_rmse = test_loss
                best_mae  = test_mae
                best_score = test_score
                best_epoch = epoch
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only MAE:{src_only_mae} \t Src_Only Score:{src_only_score}')
            print(f'After DA RMSE:{test_loss} \t After DA MAE:{test_mae} \t After DA Score:{test_score}')
            fig1 = plt.figure()
            plt.plot(pred_labels_DA, label='pred labels', linewidth=0.5)
            plt.plot(true_labels, label='true labels', linewidth=0.5)
            plt.legend()
            # fig1.savefig('training_images/'+src_id+tgt_id+'_DA.png')
            plt.close(fig1)
        
        list = [loss_weight1_m, loss_dis1_m, mean_mse_loss, mean_dis_loss, test_loss, test_mae, test_score]
        data = pd.DataFrame([list])
        data.to_csv(f"SFDA-RUL/results/loss/{src_id}_{tgt_id}_train_loss2_{hyper['lambda_w']}.csv", mode='a', header=False, index=False)

        # if epoch % 10 == 0:
        #     checkpoint1 = {'model': first_model,
        #                        'epoch': epoch,
        #                        'state_dict': first_model.state_dict()}
        #     checkpoint2 = {'model': second_model,
        #                        'epoch': epoch,
        #                        'state_dict': second_model.state_dict()}
        #     torch.save(checkpoint1, save_path+save_name1)
        #     torch.save(checkpoint2, save_path+save_name2)
    all_end_time = time.time()
    all_mins, all_secs = epoch_time(all_start_time, all_end_time)
    
    eval_start_time = time.time()
    test_loss, test_mae, test_score, pred_labels_DA, true_labels_DA = evaluate(first_model, regressor2, test_loader_t, device)
    eval_end_time = time.time()
    eval_mins, eval_secs = epoch_time(eval_start_time, eval_end_time)

    print(f'Src_Only RMSE:{src_only_loss} \t Src_Only MAE:{src_only_mae} \t Src_Only Score:{src_only_score}')
    print(f'After DA RMSE:{test_loss} \t After DA MAE:{test_mae} \t After DA Score:{test_score}')
    
    # save_name = f"SFDA-RUL/results/{src_id}_{tgt_id}.txt"
    # f = open(save_name, 'a')
    # f.write(f'Task: {src_id} -> {tgt_id} \t All Time: {all_mins}m {all_secs}s \t Eval Time: {eval_mins}m {eval_secs}s \t Best Epoch:{best_epoch} \t Best RMSE:{best_rmse} \t Best MAE:{best_mae} \t Bast Score:{best_score} \t Last Epoch:{epoch} \t Last RMSE:{test_loss} \t Last MAE:{test_mae} \t Last Score:{test_score} \t epsilon:{weight}\n')
    # f.close()
    # torch.save(first_model, save_path+'first_model.pth')
    # torch.save(second_model, save_path+'second_model.pth')
    
    return src_only_loss, test_loss


if __name__ == "__main__":
    select_method='SFDA-RUL'
    # hyper parameters
    hyper_param={ 'OC1_OC2': {'epochs':200,'batch_size':32,'lr':5e-5,'lambda_w2':0.1, 'lambda_w':0.2, 'power':0.9},
                  'OC1_OC3': {'epochs':200,'batch_size':32,'lr':1e-5,'lambda_w2':0.05, 'lambda_w':0.2, 'power':0.9},
                  'OC2_OC1': {'epochs':200,'batch_size':32,'lr':1e-5,'lambda_w2':0.1, 'lambda_w':0.1, 'power':0.9}, 
                  'OC2_OC3': {'epochs':200,'batch_size':32,'lr':1e-5,'lambda_w2':0.1, 'lambda_w':0.5, 'power':0.9},  
                  'OC3_OC1': {'epochs':200,'batch_size':32,'lr':1e-5,'lambda_w2':0.1, 'lambda_w':1, 'power':0.9},
                  'OC3_OC2': {'epochs':200,'batch_size':32,'lr':5e-5,'lambda_w2':0.1, 'lambda_w':0.5, 'power':0.9}} 
    
    layers = 3
    src_id = 'OC1'
    tgt_id = 'OC3'
    norm_id = 'None'
    data_name = 'phm_data'
    network = 'Resnet'
    gpu = False
    online = False
    
    # configuration setup
    config = get_model_config('LSTM')
    config.update({'num_runs':1, 'save':False, 'tensorboard':False,'tsne':False,'tensorboard_epoch':False, 'k_disc':1, 'k_clf':1,'iterations':1, 'layers':layers})
    
    cross_domain_train(hyper_param, device, config, ResNetFc, src_id, tgt_id, norm_id, data_name, network, gpu, online)