'''
Our Attention-based Social LSTM (AS-LSTM)
only use basic observation states [x, y]
'''
import sys
import os
import random
import copy
import time
from turtle import forward
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

model_name = 'aslstm'

class attention_scaled_dot_product(nn.Module):
    def __init__(self, scale=1.0) -> None:
        super(attention_scaled_dot_product, self).__init__()
        self.scale = scale

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None):
        '''
        forward of scaled-dot-product in attention
        q.shape = (n_batch, n_q, dim)
        k.shape = (n_batch, n_k, dim)
        v.shape = (n_batch, n_v, dim)
        mask = None: no mask
        scale = 1.0: dot-product
        dropout = 0.0: no dropout
        return: (context, softmax output) 
        '''
        att_alpha = torch.matmul(q, k.permute(0, 2, 1)) / math.sqrt(self.scale)
        if mask is not None:
            att_alpha = att_alpha.masked_fill(mask, -1e10)
        att_alpha = torch.softmax(att_alpha, dim=-1)
        context = torch.matmul(att_alpha, v)

        return context, att_alpha

class multi_head_attention(nn.Module):
    def __init__(self, in_feature, att_feature, n_head) -> None:
        super(multi_head_attention, self).__init__()
        self.in_feature = in_feature
        self.att_feature = att_feature
        self.out_feature = n_head * att_feature
        self.n_head = n_head

        self.att_Q = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.att_K = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.att_V = nn.Linear(in_features=in_feature, out_features=n_head*att_feature)
        self.dot_product = attention_scaled_dot_product(scale=att_feature)
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor=None):
        n_batch = q.shape[0]
        n_q = q.shape[1]
        n_k = k.shape[1]
        n_v = v.shape[1]
        
        vec_q = self.att_Q(q)   # n_batch x n_q x n_head*dim
        arr_K = self.att_K(k)   # n_batch x n_k x n_head*dim
        arr_V = self.att_V(v)   # n_batch x n_v x n_head*dim
        
        vec_q = vec_q.reshape(n_batch, n_q, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_q, self.att_feature)
        arr_K = arr_K.reshape(n_batch, n_k, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_k, self.att_feature)
        arr_V = arr_V.reshape(n_batch, n_v, self.n_head, self.att_feature).permute(0,2,1,3).reshape(n_batch*self.n_head, n_v, self.att_feature)
        if mask is not None:
            arr_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).reshape(n_batch*self.n_head, n_q, n_k)
        else:
            arr_mask = None

        att_output, att_alpha = self.dot_product(vec_q, arr_K, arr_V, arr_mask)
        att_output = att_output.reshape(n_batch, self.n_head, n_q, self.att_feature).permute(0,2,1,3).reshape(n_batch, n_q, self.out_feature)
        att_alpha = att_alpha.reshape(n_batch, self.n_head, n_q, n_k).permute(0,2,1,3)

        # print(q.shape, vec_q.shape, arr_K.shape, att_output.shape)

        return att_output, att_alpha

class Model_AttentionSocial_LSTM(nn.Module):
    def __init__(self, train_flag=True) -> None:
        super(Model_AttentionSocial_LSTM, self).__init__()

        self.name = model_name        
        self.model_save_dir = args['use_savedir'] + args['use_dataset'] + '/'
        self.model_save_file = self.model_save_dir + self.name + '.pt'

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.train_flag = train_flag
        self.epoch = 0

        # input embeding
        self.inp_emb_dim = 64
        self.inp_emb = nn.Linear(in_features=2, out_features=self.inp_emb_dim)
        
        # encoder-attention
        self.enc_att_num = 5
        self.enc_att_dim = 32
        self.enc_att = multi_head_attention(in_feature=self.inp_emb_dim, att_feature=self.enc_att_dim, n_head=self.enc_att_num)
        self.enc_att_out = nn.Linear(in_features=self.enc_att_num*self.enc_att_dim, out_features=self.inp_emb_dim)
        # encoder-mlp
        self.enc_mlp_upf = nn.Linear(in_features=self.inp_emb_dim, out_features=4*self.inp_emb_dim)
        self.enc_mlp_dwf = nn.Linear(in_features=4*self.inp_emb_dim, out_features=self.inp_emb_dim)

        # social-attention
        self.soc_att_num = 5
        self.soc_att_dim = 32
        self.soc_att = multi_head_attention(in_feature=self.inp_emb_dim, att_feature=self.soc_att_dim, n_head=self.soc_att_num)
        self.soc_att_out = nn.Linear(in_features=self.soc_att_num*self.soc_att_dim, out_features=self.inp_emb_dim)
        # social-mlp
        self.soc_mlp_upf = nn.Linear(in_features=self.inp_emb_dim, out_features=4*self.inp_emb_dim)
        self.soc_mlp_dwf = nn.Linear(in_features=4*self.inp_emb_dim, out_features=self.inp_emb_dim)

        # LSTM-based decoder
        self.dec_lstm = nn.LSTM(input_size=self.inp_emb_dim, hidden_size=128, batch_first=True)
        self.dec_output = nn.Linear(in_features=self.dec_lstm.hidden_size, out_features=2)

        # Activations:
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, pose_tags:Tensor, hist_tags:Tensor, pose_nghs:Tensor, hist_nghs:Tensor):

        n_batch = pose_tags.shape[0]
        n_seq = pose_tags.shape[1]
        
        tens_inp_tag = pose_tags.clone().unsqueeze(1)                                       # n_batch x 1 x n_seq x n_in_feature
        tens_inp_ngh = pose_nghs.clone()                                                    # n_batch x 4 x n_seq x n_in_feature
        tens_pos_tag = pose_tags.clone().unsqueeze(1)                                       # n_batch x 1 x n_seq x 2
        tens_pos_ngh = pose_nghs.clone()                                                    # n_batch x 4 x n_seq x 2
        tens_inp_all = torch.cat((tens_inp_tag, tens_inp_ngh), dim=1)                       # n_batch x 5 x n_seq x n_in_feature

        # input embeding
        tens_inp_emb = self.leaky_relu(self.inp_emb(tens_inp_all))                          # n_batch x 5 x n_seq x n_inp_emb_dim

        # position encoding
        t_ = np.array([np.power(3.0, 2.0*(i_//2)/self.inp_emb_dim) for i_ in range(self.inp_emb_dim)])
        tens_pe = np.array([p_/t_ for p_ in range(n_seq)])
        tens_pe[:, 0::2] = np.sin(tens_pe[:, 0::2])
        tens_pe[:, 1::2] = np.cos(tens_pe[:, 1::2])
        tens_pe = torch.from_numpy(tens_pe.copy()).float().to(args['use_device'])
        tens_pe = tens_pe.repeat(n_batch, 5, 1, 1)
        tens_inp_emb = tens_inp_emb + tens_pe                                               # n_batch x 5 x n_seq x n_inp_emb_dim

        # attention-based encoder
        tens_enc_q = tens_inp_emb[:,:,-1:,:].reshape(n_batch*5, 1, self.inp_emb_dim)        # n_batch*5 x 1       x n_inp_emb_dim
        tens_enc_k = tens_inp_emb[:,:,:-1,:].reshape(n_batch*5, n_seq-1, self.inp_emb_dim)  # n_batch*5 x n_seq-1 x n_inp_emb_dim
        tens_enc_v = tens_inp_emb[:,:,:-1,:].reshape(n_batch*5, n_seq-1, self.inp_emb_dim)  # n_batch*5 x n_seq-1 x n_inp_emb_dim

        tens_enc_att, _ = self.enc_att(tens_enc_q, tens_enc_k, tens_enc_v)                  # n_batch*5 x 1 x n_enc_att_num*n_enc_att_dim
        tens_enc_out = self.enc_att_out(tens_enc_att).reshape(n_batch, 5, self.inp_emb_dim) # n_batch x 5 x n_inp_emb_dim
        tens_enc_out = self.leaky_relu(tens_enc_out+tens_inp_emb[:,:,-1,:])
        # tens_enc_out = self.leaky_relu(tens_enc_out)

        tens_enc_mlp = self.enc_mlp_dwf(self.leaky_relu(self.enc_mlp_upf(tens_enc_out)))    # n_batch x 5 x n_inp_emb_dim
        tens_enc_mlp = self.leaky_relu(tens_enc_mlp+tens_enc_out)
        # tens_enc_out = self.leaky_relu(tens_enc_mlp)


        # attention-based social interaction
        tens_soc_q = tens_enc_mlp[:,:1,:]                                                   # n_batch x 1 x n_inp_emb_dim
        tens_soc_k = tens_enc_mlp[:,1:,:]                                                   # n_batch x 4 x n_inp_emb_dim
        tens_soc_v = tens_enc_mlp[:,1:,:]                                                   # n_batch x 4 x n_inp_emb_dim

        # attention mask
        tens_mask_ngh = torch.zeros((n_batch, 4)).int().to(args['use_device'])
        mask_ = torch.sum(torch.abs(tens_pos_ngh[:,:,-1,:]), dim=-1) < 0.1
        tens_mask_ngh = tens_mask_ngh.masked_fill(mask_, 1)
        mask_ = torch.abs(tens_pos_ngh[:,:,-1,0] - tens_pos_tag[:,0:1,-1,0]) > 50.0
        tens_mask_ngh = tens_mask_ngh.masked_fill(mask_, 1)
        tens_mask_ngh = tens_mask_ngh.bool().unsqueeze(1)

        tens_soc_att, _ = self.soc_att(tens_soc_q, tens_soc_k, tens_soc_v, tens_mask_ngh)   # n_batch x 1 x n_soc_att_num*n_soc_att_dim
        tens_soc_out = self.soc_att_out(tens_soc_att).reshape(n_batch, self.inp_emb_dim)    # n_batch x n_inp_emb_dim
        tens_soc_out = self.leaky_relu(tens_soc_out+tens_enc_mlp[:,0,:])
        # tens_soc_out = self.leaky_relu(tens_soc_out)
        
        tens_soc_mlp = self.soc_mlp_dwf(self.leaky_relu(self.soc_mlp_upf(tens_soc_out)))    # n_batch x n_inp_emb_dim
        tens_soc_mlp = self.leaky_relu(tens_soc_mlp+tens_soc_out)
        # tens_soc_mlp = self.leaky_relu(tens_soc_mlp)

        # social mask
        tens_mask_soc = torch.sum(tens_mask_ngh, dim=-1)
        tens_mask_soc = (tens_mask_soc == 4).repeat(1, self.inp_emb_dim).int().float()
        tens_soc_mlp = tens_soc_mlp * (1-tens_mask_soc)

        # LSTM-based decoder
        tens_dec_inp = tens_enc_mlp[:,0,:] + tens_soc_mlp                                       # n_batch x n_inp_emb_dim
        tens_dec_inp = tens_dec_inp.unsqueeze(-2).repeat(1, args['pred_seqlen'], 1)             # n_batch x n_predlen x n_inp_emb_dim
        tens_dec_out, (_, _) = self.dec_lstm(tens_dec_inp)
        tens_dec_out = self.dec_output(self.leaky_relu(tens_dec_out))                           # n_batch x n_predlen x 2

        # position generation
        pred_seqs = tens_pos_tag[:,0,-1:,:].repeat(1, args['pred_seqlen'], 1) + tens_dec_out    # n_batch x n_predlen x 2
                        
        return pred_seqs

lossfuc = nn.MSELoss()

def train(train_set:Dataset, valid_set:Dataset, learn_rate=0.001):
    
    # define optimizer, stopper, lossfunction
    mdl = Model_AttentionSocial_LSTM().to(args['use_device'])
    optimizer = optim.Adam(mdl.parameters(), lr=learn_rate)
    stopper = EarlyStopping(patience=100, delta=0.0)

    # load model
    if os.path.exists(mdl.model_save_file):
        mdl.load_state_dict(torch.load(mdl.model_save_file))
        _, val_loss = evaluate(mdl, valid_set)
        stopper.best_score = val_loss

    # train model
    #----------------------------------------------------------------------------------------
    for e_ in range(5000):
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

        # early-stopping
        #------------------------------------------------------------------------------------
        _, val_loss = evaluate(mdl, valid_set)
        
        if stopper.best_score is None or val_loss.item() < stopper.best_score: # save better model
            torch.save(mdl.state_dict(), mdl.model_save_file)

        stopper.check(mdl.epoch, val_loss.item())
        if stopper.early_stop:
            print("Early stop at best epoch = %d"%(stopper.best_epoch))
            break

def test(test_set:Dataset):

    mdl = Model_AttentionSocial_LSTM().to(args['use_device'])
    mdl.load_state_dict(torch.load(mdl.model_save_file))
    
    pred_test, loss_test = evaluate(mdl, test_set)
    avg_time, flops, params = runinfo(mdl, test_set)

    pred_tags = torch.cat(pred_test, dim=0).cpu().numpy()
    futu_tags = torch.cat(test_set.batches_tagfut, dim=0).cpu().numpy()
    calcError(mdl.name, pred_tags, futu_tags)

    return [p.cpu() for p in pred_test]

def evaluate(mdl:Model_AttentionSocial_LSTM, eva_set:Dataset):
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

def runinfo(mdl:Model_AttentionSocial_LSTM, eva_set:Dataset):
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