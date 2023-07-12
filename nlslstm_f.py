'''
Non-local social pooling LSTM,
reference paper:
    K. Messaoud, I. Yahiaoui, A. Verroust-Blondet & F. Nashashibi. Non-local Social Pooling for Vehicle Trajectory Prediction. 2019 IV.
'''
import sys
import os
import random
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from thop import profile
from utils import *

model_name = 'nlslstm_f'

class Model_NoLocalSocial_LSTM_F(nn.Module):
    def __init__(self, train_flag=True) -> None:
        super(Model_NoLocalSocial_LSTM_F, self).__init__()

        self.name = model_name        
        self.model_save_dir = args['use_savedir'] + args['use_dataset'] + '/'
        self.model_save_file = self.model_save_dir + self.name + '.pt'

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.train_flag = train_flag
        self.epoch = 0

        if args['use_dataset'] == 'highd':
            self.soc_map_size = (41,2)
        else:
            self.soc_map_size = (13,2)
        self.soc_map_grid = 4.0
            
        self.inp_emb = nn.Linear(in_features=6, out_features=32)
        self.enc_lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # no-local social pooling
        self.nls_att_num = 5
        self.nls_att_dim = 32
        
        self.nls_att_Q_1x1 = nn.Linear(in_features=self.enc_lstm.hidden_size, out_features=self.nls_att_num*self.nls_att_dim)
        
        self.nls_att_K_1x1 = nn.Linear(in_features=self.enc_lstm.hidden_size, out_features=self.nls_att_num*self.nls_att_dim)
        self.nls_att_K_3x2 = nn.Conv2d(in_channels=self.nls_att_num*self.nls_att_dim, out_channels=self.nls_att_num*self.nls_att_dim, kernel_size=(3,2), padding=(0,1))
        
        self.nls_att_V_1x1 = nn.Linear(in_features=self.enc_lstm.hidden_size, out_features=self.nls_att_num*self.nls_att_dim)
        self.nls_att_V_3x2 = nn.Conv2d(in_channels=self.nls_att_num*self.nls_att_dim, out_channels=self.nls_att_num*self.nls_att_dim, kernel_size=(3,2), padding=(0,1))

        self.nls_fcn = nn.Linear(in_features=self.nls_att_num*self.nls_att_dim, out_features=64)
        
        self.dec_lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dec_output = torch.nn.Linear(in_features=128, out_features=2)
        
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, pose_tags:Tensor, hist_tags:Tensor, pose_nghs:Tensor, hist_nghs:Tensor):

        n_batch = pose_tags.shape[0]
        n_seq = pose_tags.shape[1]
        
        tens_inp_tag = hist_tags.clone().unsqueeze(1)                       # n_batch x 1 x n_seq x n_in_feature
        tens_inp_ngh = hist_nghs.clone()                                    # n_batch x 4 x n_seq x n_in_feature
        tens_pos_tag = pose_tags.clone().unsqueeze(1)                       # n_batch x 1 x n_seq x 2
        tens_pos_ngh = pose_nghs.clone()                                    # n_batch x 4 x n_seq x 2
        tens_inp_all = torch.cat((tens_inp_tag, tens_inp_ngh), dim=1)       # n_batch x 5 x n_seq x n_in_feature

        # input embeding
        tens_inp_emb = self.leaky_relu(self.inp_emb(tens_inp_all))          # n_batch x 5 x n_seq x 32

        # lstm encoder
        _, (tens_enc_hid, _) = self.enc_lstm(tens_inp_emb.reshape(n_batch*5, n_seq, -1))
        tens_enc_out = tens_enc_hid.squeeze(0).reshape(n_batch, 5, -1)      # n_batch x 5 x 64

        # social encoder map
        tens_soc_map = Tensor().to(args['use_device'])                      # n_batch x (h, w) x 64
        for b_ in range(n_batch):
            sm_ = torch.zeros(size=(self.soc_map_size[0], self.soc_map_size[1], self.enc_lstm.hidden_size)).float().to(args['use_device'])
            for j_ in range(4):
                if torch.abs(torch.sum(tens_pos_ngh[b_, j_, -1, :])) < 1e-4: # [0, 0]
                    continue
                x_ = tens_pos_ngh[b_, j_, -1, 0] - tens_pos_tag[b_, 0, -1, 0]
                y_ = tens_pos_ngh[b_, j_, -1, 1]
                ix_ = (self.soc_map_size[0]-1) // 2 + round(x_.item()/self.soc_map_grid)
                iy_ = 0 if y_ < 0 else 1
                if ix_ >= 0 and ix_ < self.soc_map_size[0]:
                    sm_[ix_, iy_, :] = tens_enc_out[b_, 1+j_, :]
            
            tens_soc_map = torch.cat((tens_soc_map, sm_.unsqueeze(0)), dim=0)

        # no-local social pooling with multi-head attention
        
        vec_q = self.nls_att_Q_1x1(tens_enc_out[:,:1,:])                                        # n_batch       x 1     x nh*dh
        vec_q = vec_q.reshape(n_batch, -1, self.nls_att_num, self.nls_att_dim)                  # n_batch       x 1     x nh    x dh
        vec_q = vec_q.permute(0,2,1,3).reshape(n_batch*self.nls_att_num, -1, self.nls_att_dim)  # n_batch*nh    x 1     x dh
        
        arr_k = self.nls_att_K_1x1(tens_soc_map)                                                # n_batch       x -1    x nh*dh
        arr_k = self.nls_att_K_3x2(arr_k.permute(0,3,1,2)).permute(0,2,3,1)                     # n_batch       x -1    x nh*dh
        arr_k = arr_k.reshape(n_batch, -1, self.nls_att_num, self.nls_att_dim)                  # n_batch       x -1    x nh    x dh
        arr_k = arr_k.permute(0,2,1,3).reshape(n_batch*self.nls_att_num, -1, self.nls_att_dim)  # n_batch*nh    x -1    x dh

        arr_v = self.nls_att_V_1x1(tens_soc_map)
        arr_v = self.nls_att_V_3x2(arr_v.permute(0,3,1,2)).permute(0,2,3,1)
        arr_v = arr_v.reshape(n_batch, -1, self.nls_att_num, self.nls_att_dim)
        arr_v = arr_v.permute(0,2,1,3).reshape(n_batch*self.nls_att_num, -1, self.nls_att_dim)

        mask_ = torch.sum(torch.abs(arr_k), dim=-1) < 1e-10
        mask_ = mask_.unsqueeze(1)                                                              # n_batch*nh    x 1     x -1

        alpha = torch.matmul(vec_q, arr_k.permute(0,2,1)) / math.sqrt(self.nls_att_dim)
        alpha = alpha.masked_fill(mask_, -1e10)
        alpha = torch.softmax(alpha, dim=-1)                                                    # n_batch*nh    x 1     x -1

        tens_nls_enc = torch.matmul(alpha, arr_v)                                               # n_batch*nh    x 1     x dh
        tens_nls_enc = tens_nls_enc.reshape(n_batch, self.nls_att_num*self.nls_att_dim)         # n_batch       x nh*dh 

        tens_nls_out = self.nls_fcn(tens_nls_enc)                                           # n_batch x 64

        # lstm decoder
        tens_dec_inp = torch.cat((tens_enc_out[:,0,:], tens_nls_out), dim=-1)               # n_batch x 128
        tens_dec_inp = tens_dec_inp.repeat(args['pred_seqlen'], 1, 1).permute(1,0,2)        # n_batch x n_pred_seq x 128
        tens_dec_out, (_, _) = self.dec_lstm(tens_dec_inp)
        tens_dec_out = self.dec_output(self.leaky_relu(tens_dec_out))                       # n_batch x n_pred_seq x 2

        # position generation
        pred_seqs = tens_pos_tag[:,0,-1,:].unsqueeze(1).repeat(1,args['pred_seqlen'],1) + tens_dec_out

        return pred_seqs

lossfuc = torch.nn.MSELoss()

def train(train_set:Dataset, valid_set:Dataset, start_epoch=0, learn_rate=0.001):
    
    # define optimizer, stopper, lossfunction
    mdl = Model_NoLocalSocial_LSTM_F().to(args['use_device'])
    optimizer = optim.Adam(mdl.parameters(), lr=learn_rate)
    stopper = EarlyStopping(patience=50, delta=0.0)

    if os.path.exists(mdl.model_save_file):
        mdl.load_state_dict(torch.load(mdl.model_save_file))
        _, val_loss = evaluate(mdl, valid_set)
        stopper.best_score = val_loss

    # train model
    #----------------------------------------------------------------------------------------
    for e in range(2000):
        mdl.epoch += 1
        mdl.train_flag = True

        for k in range(train_set.total_batch_num):
            hist_tags = train_set.batches_tagpos[k].to(args['use_device'])
            hist_nghs = train_set.batches_nghpos[k].to(args['use_device'])
            pose_tags = train_set.batches_tagpos[k].to(args['use_device'])
            pose_nghs = train_set.batches_nghpos[k].to(args['use_device'])
            futu_tags = train_set.batches_tagfut[k].to(args['use_device'])

            pred_tags = mdl(pose_tags, hist_tags, pose_nghs, hist_nghs)
            loss_ = lossfuc(pred_tags, futu_tags)


            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        # validation & early-stopping
        #------------------------------------------------------------------------------------
        _, val_loss = evaluate(mdl, valid_set)
        
        if stopper.best_score is None or val_loss.item() < stopper.best_score: # save better model
            torch.save(mdl.state_dict(), mdl.model_save_file)

        stopper.check(mdl.epoch, val_loss.item())
        if stopper.early_stop:
            print("Early stop at best epoch = %d"%(stopper.best_epoch))
            break

def test(test_set:Dataset):

    mdl = Model_NoLocalSocial_LSTM_F().to(args['use_device'])
    mdl.load_state_dict(torch.load(mdl.model_save_file))
    
    pred_test, loss_test = evaluate(mdl, test_set)
    avg_time, flops, params = runinfo(mdl, test_set)

    pred_tags = torch.cat(pred_test, dim=0).cpu().numpy()
    futu_tags = torch.cat(test_set.batches_tagfut, dim=0).cpu().numpy()
    calcError(mdl.name, pred_tags, futu_tags)

    return [p.cpu() for p in pred_test]

def evaluate(mdl:Model_NoLocalSocial_LSTM_F, eva_set:Dataset):
    eva_pred_tags = []
    eva_loss = 0
    cnt_samples = 0
    mdl.train_flag = False
    for k in range(eva_set.total_batch_num):
        pose_tags = eva_set.batches_tagpos[k].to(args['use_device'])
        hist_tags = eva_set.batches_tagsta[k].to(args['use_device'])
        pose_nghs = eva_set.batches_nghpos[k].to(args['use_device'])
        hist_nghs = eva_set.batches_nghsta[k].to(args['use_device'])
        futu_tags = eva_set.batches_tagfut[k].to(args['use_device'])

        with torch.no_grad():
            pred_tags = mdl(pose_tags, hist_tags, pose_nghs, hist_nghs)
            
            eva_pred_tags.append(pred_tags)
            loss_ = lossfuc(pred_tags, futu_tags)

            eva_loss = eva_loss + loss_ * pose_tags.shape[0]
            cnt_samples = cnt_samples + pose_tags.shape[0]
    
    eva_loss = eva_loss / cnt_samples
    return eva_pred_tags, eva_loss

def runinfo(mdl:Model_NoLocalSocial_LSTM_F, eva_set:Dataset):
    avg_time = 0
    mdl.train_flag = False
    pose_tags = eva_set.batches_tagpos[0].to(args['use_device'])
    hist_tags = eva_set.batches_tagsta[0].to(args['use_device'])
    pose_nghs = eva_set.batches_nghpos[0].to(args['use_device'])
    hist_nghs = eva_set.batches_nghsta[0].to(args['use_device'])
    with torch.no_grad():
        for i in range(hist_tags.shape[0]):
            t1 = time.time()
            mdl(pose_tags[i:i+1], hist_tags[i:i+1], pose_nghs[i:i+1], hist_nghs[i:i+1])
            t2 = time.time()
            avg_time += (t2-t1)
    
    avg_time = avg_time / hist_tags.shape[0]
    flops, params = profile(mdl, inputs=(pose_tags[:1], hist_tags[:1], pose_nghs[:1], hist_nghs[:1],))
    flops = flops * 2
    print("AvgTime = %.2fms, FLOPs = %.2fM, Params = %.2fM"%(avg_time*1e3, flops/1e6, params/1e6))
            
    return avg_time, flops, params

if __name__ == '__main__':

    print("use device: %s"%args['use_device'])
    dir_pkl = './data_pkl/' + args['use_dataset'] + '/'

    with open(dir_pkl+'train.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open(dir_pkl+'valid.pkl', 'rb') as f:
        valid_set = pickle.load(f)
    with open(dir_pkl+'test.pkl', 'rb') as f:
        test_set = pickle.load(f)

    if args['is_train']:
        train(train_set, valid_set, learn_rate=0.001)

    test(test_set)