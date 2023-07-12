'''
Constant Velocity Model Prediction, only use basic observation states [x, y, v]
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

model_name = 'cvm'

class Model_ConstantVelocity(nn.Module):
    def __init__(self) -> None:
        super(Model_ConstantVelocity, self).__init__()

        self.name = model_name
        print("model name is : %s"%self.name)

    def forward(self, pose_tags:Tensor):
        vel_seqs = (pose_tags[:,-1,:] - pose_tags[:,-2,:]).unsqueeze(1).repeat(1, args['pred_seqlen'], 1)
        pos_seqs = pose_tags[:,-1,:].unsqueeze(1).repeat(1, args['pred_seqlen'], 1)
        ts = torch.linspace(1, args['pred_seqlen'], args['pred_seqlen']).to(args['use_device']).reshape(1, args['pred_seqlen'], 1).repeat(pos_seqs.shape[0], 1, 2)
        pred_seqs = pos_seqs + vel_seqs * ts

        return pred_seqs

lossfuc = nn.MSELoss()

def test(test_set:Dataset):
    
    mdl = Model_ConstantVelocity().to(args['use_device'])

    pred_test, loss_test = evaluate(mdl, test_set)
    avg_time, flops, params = runinfo(mdl, test_set)

    pred_tags = torch.cat(pred_test, dim=0).cpu().numpy()
    futu_tags = torch.cat(test_set.batches_tagfut, dim=0).cpu().numpy()
    calcError(mdl.name, pred_tags, futu_tags)

    return [p.cpu() for p in pred_test]

def evaluate(mdl:Model_ConstantVelocity, eva_set:Dataset):

    eva_pred_tags = []
    eva_loss = 0
    cnt_samples = 0
    for k in range(eva_set.total_batch_num):
        pose_tags = eva_set.batches_tagpos[k].to(args['use_device'])
        futu_tags = eva_set.batches_tagfut[k].to(args['use_device'])

        pred_tags = mdl(pose_tags)
        loss_ = lossfuc(pred_tags, futu_tags)
        eva_pred_tags.append(pred_tags)
        eva_loss = eva_loss + loss_ * pose_tags.shape[0]
        cnt_samples = cnt_samples + pose_tags.shape[0]
    
    eva_loss = eva_loss / cnt_samples
    return eva_pred_tags, eva_loss

def runinfo(mdl:Model_ConstantVelocity, eva_set:Dataset):
    avg_time = 0
    hist_tags = eva_set.batches_tagpos[0].to(args['use_device'])
    with torch.no_grad():
        for i in range(hist_tags.shape[0]):
            t1 = time.time()
            mdl(hist_tags[i:i+1])
            t2 = time.time()
            avg_time += (t2-t1)
    
    avg_time = avg_time / hist_tags.shape[0]
    flops, params = profile(mdl, inputs=(hist_tags[:1],))
    flops = flops * 2
    print("AvgTime = %.2fms, FLOPs = %.2fM, Params = %.2fM"%(avg_time*1e3, flops/1e6, params/1e6))
            
    return avg_time, flops, params

if __name__ == '__main__':

    print("use device: %s"%args['use_device'])
    dir_pkl = './data_pkl/' + args['use_dataset'] + '/'

    with open(dir_pkl+'test.pkl', 'rb') as f:
        test_set = pickle.load(f)
    test(test_set)    