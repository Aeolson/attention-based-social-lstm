'''
Maneuver-based LSTM,
reference paper:
    N. Deo & M. M. Trivedi. Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs. 2018 IV.
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

model_name = 'mlstm'

class Model_ManeuverLSTM(nn.Module):
    def __init__(self, train_flag=True) -> None:
        super(Model_ManeuverLSTM, self).__init__()

        self.name = model_name
        self.model_save_dir = args['use_savedir'] + args['use_dataset'] + '/'
        self.model_save_file = self.model_save_dir + self.name + '.pt'
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.train_flag = train_flag
        self.epoch = 0

        self.mod_emb = nn.Linear(in_features=2*5, out_features=64)
        self.mod_lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.mod_latout = nn.Linear(in_features=128, out_features=2)
        self.mod_lonout = nn.Linear(in_features=128, out_features=2)

        self.enc_emb = nn.Linear(in_features=2*5, out_features=64)
        self.enc_lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dec_lstm = nn.LSTM(input_size=132, hidden_size=128, batch_first=True)
        self.dec_output = nn.Linear(in_features=128, out_features=2)
        
        # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hist_stas:Tensor, pose_tags:Tensor, mode_enc1:Tensor=None, mode_enc2:Tensor=None):
        
        n_batch = hist_stas.shape[0]
        n_seq = hist_stas.shape[1]
        
        # transform list into Tensor
        tens_inp_sta = hist_stas.clone()               # n_batch x n_seq x n_in_feature
        tens_pos_tag = pose_tags.clone()               # n_batch x n_seq x 2
        if self.train_flag:
            tens_mod_lat = mode_enc1.clone()           # n_batch x 2
            tens_mod_lon = mode_enc2.clone()           # n_batch x 2

        # lstm maneuver classifier
        tens_mod_emb = self.leaky_relu(self.mod_emb(tens_inp_sta))  # n_batch x 64
        _, (tens_mod_hid, _) = self.mod_lstm(tens_mod_emb)
        tens_mod_out = tens_mod_hid.squeeze(0)                      # n_batch x 128
        tens_lat_out = self.softmax(self.mod_latout(tens_mod_out))  # n_batch x 2
        tens_lon_out = self.softmax(self.mod_lonout(tens_mod_out))  # n_batch x 2

        # lstm encoder
        tens_enc_emb = self.leaky_relu(self.enc_emb(tens_inp_sta))  # n_batch x 64
        _, (tens_enc_hid, _) = self.enc_lstm(tens_enc_emb)
        tens_enc_out = tens_enc_hid.squeeze(0)                      # n_batch x 128

        # lstm decoder
        if self.train_flag: # train stage
            tens_dec_inp = torch.cat((tens_enc_out, tens_mod_lat, tens_mod_lon), dim=-1)    # n_batch x 132
            tens_dec_inp = tens_dec_inp.unsqueeze(1).repeat(1, args['pred_seqlen'], 1)      # n_batch x n_predlen x 132
            tens_dec_out, (_, _) = self.dec_lstm(tens_dec_inp)
            tens_dec_out = self.dec_output(tens_dec_out)                                    # n_batch x n_predlen x 2
            
            # position generation
            pred_seqs = tens_pos_tag[:,-1:,:].repeat(1, args['pred_seqlen'], 1) + tens_dec_out

        else: # test stage
            # maneuver mode encoding
            mod_enc_ = Tensor().to(args['use_device'])                                      # 4 x 4
            for i in range(2):
                for j in range(2):
                    lat_ = torch.zeros(size=(2,)).to(args['use_device'])
                    lat_[i] = 1.
                    lon_ = torch.zeros(size=(2,)).to(args['use_device'])
                    lon_[j] = 1.
                    enc_ = torch.cat((lat_, lon_), dim=-1)
                    mod_enc_ = torch.cat((mod_enc_, enc_.unsqueeze(0)), dim=0)
            mod_enc_ = mod_enc_.unsqueeze(0).repeat(n_batch, 1, 1)                          # n_batch x 4 x 4
            
            tens_dec_inp = tens_enc_out.unsqueeze(1).repeat(1, 4, 1)                        # n_batch x 4 x 128
            tens_dec_inp = torch.cat((tens_dec_inp, mod_enc_), dim=-1)                      # n_batch x 4 x 132

            tens_dec_inp = tens_dec_inp.unsqueeze(2).repeat(1, 1, args['pred_seqlen'], 1)                   # n_batch x 4 x n_predlen x 132
            tens_dec_out, (_, _) = self.dec_lstm(tens_dec_inp.reshape(n_batch*4, args['pred_seqlen'], -1))
            tens_dec_out = self.dec_output(tens_dec_out.reshape(n_batch, 4, args['pred_seqlen'], -1))       # n_batch x 4 x n_predlen x 2
            
            # position generation
            pred_seqs = tens_pos_tag[:,-1:,:].unsqueeze(1).repeat(1, 4, args['pred_seqlen'], 1) + tens_dec_out
        
        return pred_seqs, tens_lat_out, tens_lon_out

lossfuc_mse = nn.MSELoss()
lossfuc_bce = nn.BCELoss()

def train(train_set:Dataset, valid_set:Dataset, learn_rate=0.001):
    
    mdl = Model_ManeuverLSTM().to(args['use_device'])
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
            hist_tags = train_set.batches_tagpos[k].unsqueeze(1)
            hist_nghs = train_set.batches_nghpos[k]
            hist_obss = torch.cat((hist_tags, hist_nghs), dim=1).permute(0,2,1,3).reshape(hist_tags.shape[0], hist_tags.shape[2], -1).to(args['use_device'])
            pose_tags = train_set.batches_tagpos[k].to(args['use_device'])
            futu_tags = train_set.batches_tagfut[k].to(args['use_device'])
            latm_encs = train_set.batches_latenc[k].to(args['use_device'])
            lonm_encs = train_set.batches_lonenc[k].to(args['use_device'])
                
            pred_tags, prob_lats, prob_lons = mdl(hist_obss, pose_tags, latm_encs, lonm_encs)
            loss_pred = lossfuc_mse.forward(pred_tags, futu_tags)
            loss_mode = lossfuc_bce.forward(prob_lats, latm_encs) + lossfuc_bce.forward(prob_lons, lonm_encs)
            loss_ = loss_pred + loss_mode
          
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
    
    mdl = Model_ManeuverLSTM().to(args['use_device'])
    mdl.load_state_dict(torch.load(mdl.model_save_file))

    pred_test, loss_test = evaluate(mdl, test_set)
    avg_time, flops, params = runinfo(mdl, test_set)

    pred_tags = torch.cat(pred_test, dim=0).cpu().numpy()
    futu_poss = torch.cat(test_set.batches_tagfut, dim=0).cpu().numpy()
    calcError(mdl.name, pred_tags, futu_poss)

    return [p.cpu() for p in pred_test]

def evaluate(mdl:Model_ManeuverLSTM, eva_set:Dataset):

    eva_pred_tags = []
    eva_loss = 0
    cnt_samples = 0
    mdl.train_flag = False
    for k in range(eva_set.total_batch_num):
        hist_tags = eva_set.batches_tagpos[k].unsqueeze(1)
        hist_nghs = eva_set.batches_nghpos[k]
        hist_obss = torch.cat((hist_tags, hist_nghs), dim=1).permute(0,2,1,3).reshape(hist_tags.shape[0], hist_tags.shape[2], -1).to(args['use_device'])
        pose_tags = eva_set.batches_tagpos[k].to(args['use_device'])
        futu_tags = eva_set.batches_tagfut[k].to(args['use_device'])
        with torch.no_grad():
            pred_, lats_, lons_ = mdl(hist_obss, pose_tags, None, None)
            best_pred = []
            for b_ in range(eva_set.batches_tagpos[k].shape[0]):
                b_p = 0
                b_k = 0
                for i_ in range(2):
                    for j_ in range(2):
                        if lats_[b_,i_] * lons_[b_,j_] > b_p:
                            b_p = lats_[b_,i_] * lons_[b_,j_]
                            b_k = i_ * 2 + j_
                best_pred.append(pred_[b_, b_k, :, :])
            
            pred_tags = torch.stack(best_pred)
            eva_pred_tags.append(pred_tags)
            loss_ = lossfuc_mse(pred_tags, futu_tags)

            eva_loss = eva_loss + loss_ * pose_tags.shape[0]
            cnt_samples = cnt_samples + pose_tags.shape[0]
    
    eva_loss = eva_loss / cnt_samples    
    return eva_pred_tags, eva_loss

def runinfo(mdl:Model_ManeuverLSTM, eva_set:Dataset):
    avg_time = 0
    mdl.train_flag = False
    hist_tags = eva_set.batches_tagpos[0].unsqueeze(1)
    hist_nghs = eva_set.batches_nghpos[0]
    hist_obss = torch.cat((hist_tags, hist_nghs), dim=1).permute(0,2,1,3).reshape(hist_tags.shape[0], hist_tags.shape[2], -1).to(args['use_device'])
    pose_tags = eva_set.batches_tagpos[0].to(args['use_device'])
    with torch.no_grad():
        for i in range(hist_tags.shape[0]):
            t1 = time.time()
            mdl(hist_obss[i:i+1], pose_tags[i:i+1])
            t2 = time.time()
            avg_time += (t2-t1)
    
    avg_time = avg_time / hist_tags.shape[0]
    flops, params = profile(mdl, inputs=(hist_obss[:1], pose_tags[:1], ))
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

    is_train = 0
    if args['is_train']:
        train(train_set, valid_set, learn_rate=0.001)

    test(test_set)