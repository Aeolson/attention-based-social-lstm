'''
Vanilla LSTM with encoder-decoder structure (seq2seq),
both encoder and decoder are developed with single-layer LSTM network,
pipeline is:
    [input] -----> input_embeding -----> lstm_encoder -----> lstm_decoder -----> output_fcn -----> [output]
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

model_name = 'vlstm_f'

class Model_VanillaLSTM_F(nn.Module):
    def __init__(self, train_flag=True) -> None:
        super(Model_VanillaLSTM_F, self).__init__()

        self.name = model_name        
        self.model_save_dir = args['use_savedir'] + args['use_dataset'] + '/'
        self.model_save_file = self.model_save_dir + self.name + '.pt'

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.train_flag = train_flag
        self.epoch = 0
        
        self.inupt_emb = nn.Linear(in_features=6, out_features=64)
        self.encoder_lstm = nn.LSTM(input_size=self.inupt_emb.out_features, hidden_size=64, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=self.encoder_lstm.hidden_size, hidden_size=128, batch_first=True)
        self.decoder_output = nn.Linear(in_features=self.decoder_lstm.hidden_size, out_features=2)
                
        # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, pose_tags:Tensor, hist_tags:Tensor, pose_nghs:Tensor, hist_nghs:Tensor):

        n_batch = pose_tags.shape[0]
        n_seq = pose_tags.shape[1]
        
        # transform list into Tensor
        tens_inp_sta = hist_tags.clone()                                                # n_batch x n_seq x n_in_feature
        tens_pos_tag = pose_tags.clone()                                                # n_batch x n_seq x 2

        # input embeding
        tens_inp_emb = self.leaky_relu(self.inupt_emb(tens_inp_sta))

        # lstm encoder
        _, (tens_enc_hid, _) = self.encoder_lstm(tens_inp_emb)
        tens_enc_hid = tens_enc_hid.squeeze(0)                                          # n_batch x n_lstm_hid_dim
        # lstm decoder
        tens_dec_inp = tens_enc_hid.unsqueeze(1).repeat(1, args['pred_seqlen'], 1)      # n_batch x n_prelen x n_enc_out_dim
        tens_dec_out, (_, _) = self.decoder_lstm(tens_dec_inp)
        tens_dec_out = self.decoder_output(self.leaky_relu(tens_dec_out))               # n_batch x n_predlen x 2

        # position generation
        pred_seqs = tens_pos_tag[:,-1:,:].repeat(1, args['pred_seqlen'], 1) + tens_dec_out

        return pred_seqs

lossfuc = torch.nn.MSELoss()

def train(train_set:Dataset, valid_set:Dataset, learn_rate=0.01):

    mdl = Model_VanillaLSTM_F().to(args['use_device'])
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
            pose_tags = train_set.batches_tagpos[k].to(args['use_device'])
            hist_tags = train_set.batches_tagsta[k].to(args['use_device'])
            pose_nghs = train_set.batches_nghpos[k].to(args['use_device'])
            hist_nghs = train_set.batches_nghsta[k].to(args['use_device'])
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

    mdl = Model_VanillaLSTM_F().to(args['use_device'])
    mdl.load_state_dict(torch.load(mdl.model_save_file))
    
    pred_test, loss_test = evaluate(mdl, test_set)
    avg_time, flops, params = runinfo(mdl, test_set)

    pred_tags = torch.cat(pred_test, dim=0).cpu().numpy()
    futu_tags = torch.cat(test_set.batches_tagfut, dim=0).cpu().numpy()
    calcError(mdl.name, pred_tags, futu_tags)

    return [p.cpu() for p in pred_test]

def evaluate(mdl:Model_VanillaLSTM_F, eva_set:Dataset):
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

def runinfo(mdl:Model_VanillaLSTM_F, eva_set:Dataset):
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