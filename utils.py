import os
import sys
import glob
import math
import pandas as pd
import numpy as np
from sklearn import datasets
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

args = {}
args['use_device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['use_dataset'] = 'ngsim'
# args['use_dataset'] = 'highd'
args['use_savedir'] = './model_save/'
args['is_train'] = True

args['fps'] = 5
args['hist_ts'] = 3.0
args['pred_ts'] = 5.0
args['hist_seqlen'] = int(args['hist_ts'] * args['fps'])
args['pred_seqlen'] = int(args['pred_ts'] * args['fps'])

data_test_flag = 0
draw_traj_flag = 0

''' Dataset class for the NGSIM dataset '''
class Dataset(object):

    def __init__(self, data_name:str, data_type:str, batch_size=0) -> None:
        
        self.dataset_name = data_name
        self.dataset_type = data_type
        self.file_dir = './my_dataset/' + data_name + '/' + data_type + '/'

        if not os.path.exists(self.file_dir):
            print("Dataset is wrong !!!!")
            sys.exit()

        self.batch_size = batch_size
        self.fps = args['fps']
        self.hist_len = args['hist_seqlen']
        self.pred_len = args['pred_seqlen']
        self.hist_maxlen = args['hist_seqlen']
        
        self.lat_mode_num = 2
        self.lon_mode_num = 2

        self.batches_tagpos = []
        self.batches_tagsta = []
        self.batches_nghpos = []
        self.batches_nghsta = []
        
        self.batches_tagfut = []

        self.batches_latmod = []
        self.batches_latenc = []
        self.batches_lonmod = []
        self.batches_lonenc = []

        self.total_batch_num = 0
        self.total_sample_num = 0
    
    def run(self):
        if self.batch_size == 0:
            self.run_insingle()
        else:
            self.run_inbatch()

    def run_inbatch(self):
        allfiles = glob.glob(self.file_dir + '/*.csv')
        print("Total trajectory files: %d"%len(allfiles))

        b_tagpos = []
        b_tagsat = []
        b_nghpos = []
        b_nghsta = []
        b_tagfut = []
        b_latmod = []
        b_latenc = []
        b_lonmod = []
        b_lonenc = []

        for f_ in allfiles:
            
            if data_test_flag > 0:
                if len(self.batches_tagpos) >= 50:
                    break

            df = pd.read_csv(f_)
            cutin_idx = self.getCutinIdx(df)
            n_idx = cutin_idx
            while n_idx > 0:
                pos_tag, sta_tag, pos_ngh, sta_ngh = self.getHistory(df, n_idx)
                fut_tag = self.getPredict(df, n_idx)

                if pos_tag is not None and fut_tag is not None:
                    
                    if draw_traj_flag > 0:
                        self.draw_trj(pos_tag, fut_tag, pos_ngh)

                    lat_mod, lon_mod, lat_enc, lon_enc = self.getModes(pos_tag, fut_tag)

                    b_tagpos.append(pos_tag)
                    b_tagsat.append(sta_tag)
                    b_nghpos.append(pos_ngh)
                    b_nghsta.append(sta_ngh)
                    b_tagfut.append(fut_tag)
                    b_latmod.append(lat_mod)
                    b_latenc.append(lat_enc)
                    b_lonmod.append(lon_mod)
                    b_lonenc.append(lon_enc)
                    
                
                if self.batch_size > 0 and len(b_tagpos) == self.batch_size:
                    self.batches_tagpos.append(torch.stack(b_tagpos))
                    self.batches_tagsta.append(torch.stack(b_tagsat))
                    self.batches_nghpos.append(torch.stack(b_nghpos))
                    self.batches_nghsta.append(torch.stack(b_nghsta))
                    self.batches_tagfut.append(torch.stack(b_tagfut))

                    self.batches_latenc.append(torch.stack(b_latenc))
                    self.batches_lonenc.append(torch.stack(b_lonenc))
                    self.batches_latmod.append(torch.Tensor(b_latmod))
                    self.batches_lonmod.append(torch.Tensor(b_lonmod))

                    b_tagpos = []
                    b_tagsat = []
                    b_nghpos = []
                    b_nghsta = []
                    b_tagfut = []
                    b_latmod = []
                    b_latenc = []
                    b_lonmod = []
                    b_lonenc = []

                if self.dataset_name == 'ngsim':
                    n_idx -= int(1.0 * 10)
                elif self.dataset_name == 'highd':
                    n_idx -= int(1.0 * 25)
                else:
                    print("dataname error !!!")
                    sys.exit()

        if len(b_tagpos) > 10:
            self.batches_tagpos.append(torch.stack(b_tagpos))
            self.batches_tagsta.append(torch.stack(b_tagsat))
            self.batches_nghpos.append(torch.stack(b_nghpos))
            self.batches_nghsta.append(torch.stack(b_nghsta))
            self.batches_tagfut.append(torch.stack(b_tagfut))

            self.batches_latenc.append(torch.stack(b_latenc))
            self.batches_lonenc.append(torch.stack(b_lonenc))
            self.batches_latmod.append(torch.Tensor(b_latmod))
            self.batches_lonmod.append(torch.Tensor(b_lonmod))


        self.total_batch_num = len(self.batches_tagpos)
        self.total_sample_num = sum([len(b) for b in self.batches_tagpos])
        print("Total number: batches %3d, samples %6d"%(self.total_batch_num, self.total_sample_num))
    
    def run_insingle(self):
        allfiles = glob.glob(self.file_dir + '/*.csv')
        print("Total trajectory files: %d"%len(allfiles))

        for f_ in allfiles:
            
            if data_test_flag > 0:
                if len(self.batches_tagpos) >= 50:
                    break
            
            b_tagpos = []
            b_tagsat = []
            b_nghpos = []
            b_nghsta = []
            b_tagfut = []
            b_latmod = []
            b_latenc = []
            b_lonmod = []
            b_lonenc = []

            df = pd.read_csv(f_)
            cutin_idx = self.getCutinIdx(df)
            n_idx = cutin_idx
            while n_idx > 0:
                pos_tag, sta_tag, pos_ngh, sta_ngh = self.getHistory(df, n_idx)
                fut_tag = self.getPredict(df, n_idx)

                if pos_tag is None or fut_tag is None:
                    break
                    
                lat_mod, lon_mod, lat_enc, lon_enc = self.getModes(pos_tag, fut_tag)

                b_tagpos.append(pos_tag)
                b_tagsat.append(sta_tag)
                b_nghpos.append(pos_ngh)
                b_nghsta.append(sta_ngh)
                b_tagfut.append(fut_tag)
                b_latmod.append(lat_mod)
                b_latenc.append(lat_enc)
                b_lonmod.append(lon_mod)
                b_lonenc.append(lon_enc)
                

                if self.dataset_name == 'ngsim':
                    n_idx -= int(1.0 * 10)
                elif self.dataset_name == 'highd':
                    n_idx -= int(1.0 * 25)
                else:
                    print("dataname error !!!")
                    sys.exit()

            if len(b_tagpos) > 0:
                self.batches_tagpos.append(torch.stack(b_tagpos))
                self.batches_tagsta.append(torch.stack(b_tagsat))
                self.batches_nghpos.append(torch.stack(b_nghpos))
                self.batches_nghsta.append(torch.stack(b_nghsta))
                self.batches_tagfut.append(torch.stack(b_tagfut))

                self.batches_latenc.append(torch.stack(b_latenc))
                self.batches_lonenc.append(torch.stack(b_lonenc))
                self.batches_latmod.append(torch.Tensor(b_latmod))
                self.batches_lonmod.append(torch.Tensor(b_lonmod))

        self.total_batch_num = len(self.batches_tagpos)
        self.total_sample_num = sum([len(b) for b in self.batches_tagpos])
        print("Total number: batches %3d, samples %6d"%(self.total_batch_num, self.total_sample_num))

    def getCutinIdx(self, df:pd.DataFrame):
        return df[df['Frame_ID'] == -1].index.tolist()[0]

    def getHistory(self, df:pd.DataFrame, n_idx:int):

        if df.loc[n_idx, 'Ego_ID'] == 0: # no ego
            return None, None, None, None
        
        if self.dataset_name == 'ngsim':
            down_step = 10.0 / self.fps # ngsim: 10Hz -> self.fps
        elif self.dataset_name == 'highd':
            down_step = 25.0 / self.fps # highd: 25Hz -> self.fps

        ref_x = df.loc[n_idx, 'Tag_X']
        ref_y = df.loc[n_idx, 'Merge_Y']
        cut_h = np.sign(df.loc[n_idx, 'Lane_To'] - df.loc[n_idx, 'Lane_From'])

        posseq_tag = [];    staseq_tag = []
        posseq_ego = [];    staseq_ego = []
        posseq_egopre = []; staseq_egopre = []
        posseq_tagfol = []; staseq_tagfol = []
        posseq_tagpre = []; staseq_tagpre = []


        for k in range(self.hist_maxlen):
            idx = float(n_idx) - k*down_step
            idx_b = math.floor(idx)

            if idx_b < 0:
                break
            if df.loc[idx_b, 'Ego_ID'] != df.loc[n_idx, 'Ego_ID']:
                break

            # tag position & state
            pos_t, sta_t = self.getStateVector(df, idx, name='Tag', ref=(ref_x, ref_y, cut_h))
            posseq_tag.insert(0, pos_t)
            staseq_tag.insert(0, sta_t)

            # ego position & state
            pos_e, sta_e = self.getStateVector(df, idx, name='Ego', ref=(ref_x, ref_y, cut_h))
            posseq_ego.insert(0, pos_e)
            staseq_ego.insert(0, sta_e)

            # egopre position & state
            if df.loc[idx_b, 'EgoPre_ID'] != 0:
                pos_ep, sta_ep = self.getStateVector(df, idx, name='EgoPre', ref=(ref_x, ref_y, cut_h))
            else:
                pos_ep = [0, 0]
                sta_ep = [0, 0, 0, 0, 0, 0]
            posseq_egopre.insert(0, pos_ep)
            staseq_egopre.insert(0, sta_ep)
            
            # tagfol position & state
            if df.loc[idx_b, 'TagFol_ID'] != 0:
                pos_tf, sta_tf = self.getStateVector(df, idx, name='TagFol', ref=(ref_x, ref_y, cut_h))
            else:
                pos_tf = [0, 0]
                sta_tf = [0, 0, 0, 0, 0, 0]
            posseq_tagfol.insert(0, pos_tf)
            staseq_tagfol.insert(0, sta_tf)
            
            # tag position & state
            if df.loc[idx_b, 'TagPre_ID'] != 0:
                pos_tp, sta_tp = self.getStateVector(df, idx, name='TagPre', ref=(ref_x, ref_y, cut_h))
            else:
                pos_tp = [0, 0]
                sta_tp = [0, 0, 0, 0, 0, 0]
            posseq_tagpre.insert(0, pos_tp)
            staseq_tagpre.insert(0, sta_tp)

        if len(posseq_tag) < self.hist_len:
            return None, None, None, None

        # transform the list into Tensor
        posseq_tag = Tensor(posseq_tag).float()
        staseq_tag = Tensor(staseq_tag).float()

        posseq_ngh = Tensor([posseq_ego, posseq_egopre, posseq_tagfol, posseq_tagpre]).float()
        staseq_ngh = Tensor([staseq_ego, staseq_egopre, staseq_tagfol, staseq_tagpre]).float()
        
        return posseq_tag, staseq_tag, posseq_ngh, staseq_ngh

    def getPredict(self, df:pd.DataFrame, n_idx:int):

        if self.dataset_name == 'ngsim':
            down_step = 10.0 / self.fps # ngsim: 10Hz -> self.fps
        elif self.dataset_name == 'highd':
            down_step = 25.0 / self.fps # highd: 25Hz -> self.fps

        ref_x = df.loc[n_idx, 'Tag_X']
        ref_y = df.loc[n_idx, 'Merge_Y']
        cut_h = np.sign(df.loc[n_idx, 'Lane_To'] - df.loc[n_idx, 'Lane_From'])

        posseq_tag = []

        for k in range(self.pred_len):
            idx = float(n_idx) + (k+1)*down_step
            idx_t = math.ceil(idx)

            if idx_t >= len(df):
                break
            
            pos_, sta_ = self.getStateVector(df, idx, name='Tag', ref=(ref_x, ref_y, cut_h))
            posseq_tag.append(pos_)

        if len(posseq_tag) < self.pred_len:
            return None
        
        posseq_tag = Tensor(posseq_tag).float()
        
        return posseq_tag

    def getModes(self, hist_pos:Tensor, pred_pos:Tensor):

        # lateral modes
        tag_final_y = pred_pos[-1, 1].item()
        if self.lat_mode_num == 2:
            if tag_final_y > 0:
                lat_mod = 1
            else:
                lat_mod = 0
        elif self.lat_mode_num == 3:
            if tag_final_y > 0.5:
                lat_mod = 2
            elif tag_final_y > -0.5:
                lat_mod = 1
            else:
                lat_mod = 0
        else:
            lat_mod = 0 # no multi-mode
        
        # longitudinal modes
        avg_vh = (hist_pos[-1,0] - hist_pos[0,0]) / (self.hist_len - 1.0)
        avg_vp = (pred_pos[-1,0] - pred_pos[0,0]) / (self.pred_len - 1.0)
        if self.lon_mode_num == 2:
            if avg_vp < 0.8 * avg_vh:
                lon_mod = 1
            else:
                lon_mod = 0
        elif self.lon_mode_num == 3:
            if avg_vp > 1.2 * avg_vh:
                lon_mod = 2
            elif avg_vp < 0.8 * avg_vh:
                lon_mod = 1
            else:
                lon_mod = 0
        else:
            lon_mod = 0

        lat_enc = torch.zeros(size=(self.lat_mode_num,)).float()
        lon_enc = torch.zeros(size=(self.lon_mode_num,)).float()
        lat_enc[lat_mod] = 1.
        lon_enc[lon_mod] = 1.

        return lat_mod, lon_mod, lat_enc, lon_enc

    def getStateVector(self, df:pd.DataFrame, idx:float, name:str, ref:tuple):

        if abs(idx - round(idx)) < 0.001:
            idx = round(idx)

            x_   = df.loc[idx, name+'_X'] - ref[0]
            y_   = df.loc[idx, name+'_Y'] - ref[1]
            y_   = y_ * ref[2]

            hws_ = df.loc[idx, name+'_X'] - df.loc[idx, 'Tag_X']
            vel_ = df.loc[idx, name+'_Vel']
            acc_ = df.loc[idx, name+'_Acc']
            wid_ = df.loc[idx, name+'_Wid']
            len_ = df.loc[idx, name+'_Len']
        else:
            alpha = float(idx) - float(math.floor(idx))
            idx = math.floor(idx)

            x_   = (1-alpha) * (df.loc[idx, name+'_X'] - ref[0])                + alpha * (df.loc[idx+1, name+'_X'] - ref[0])
            y_   = (1-alpha) * (df.loc[idx, name+'_Y'] - ref[1])                + alpha * (df.loc[idx+1, name+'_Y'] - ref[1])
            y_   = y_ * ref[2]

            hws_ = (1-alpha) * (df.loc[idx, name+'_X'] - df.loc[idx, 'Tag_X'])  + alpha * (df.loc[idx+1, name+'_X'] - df.loc[idx, 'Tag_X'])
            vel_ = (1-alpha) * (df.loc[idx, name+'_Vel'])                       + alpha * (df.loc[idx+1, name+'_Vel'])
            acc_ = (1-alpha) * (df.loc[idx, name+'_Acc'])                       + alpha * (df.loc[idx+1, name+'_Acc'])
            wid_ = df.loc[idx, name+'_Wid']
            len_ = df.loc[idx, name+'_Len']     
        
        return [x_, y_], [hws_, y_, vel_, acc_, wid_, len_]

    def draw_trj(self, pos_tag:Tensor, fut_tag:Tensor, pos_ngh:Tensor):

        plt.figure()
        plt.plot(pos_tag[:,0].numpy(), pos_tag[:,1].numpy(), '-b', marker='o', lw=2)
        plt.plot(fut_tag[:,0].numpy(), fut_tag[:,1].numpy(), '-r', marker='x', lw=2)

        for i_ngh in range(4):
            if i_ngh == 0: #ego
                plt.plot(pos_ngh[0,:,0].numpy(), pos_ngh[0,:,1].numpy(), '-g', marker='o', lw=2)
            else:
                p_ = []
                for k in range(pos_ngh.shape[1]):
                    if abs(pos_ngh[i_ngh,k,0] - pos_tag[k,0]) < 200:
                        p_.append(pos_ngh[i_ngh, k, :].numpy())
                p_ = np.array(p_)
                if len(p_) > 0:
                    plt.plot(p_[:,0], p_[:,1], '-k', marker='o', lw=2)
        
        plt.show()

'''  EarlyStopping class for training '''
class EarlyStopping(object):
    
    def __init__(self, patience=5, verbose=False, delta=0):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True, 为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def check(self, epoch, val_loss):

        score = val_loss

        if self.best_epoch is None or score < self.best_score - self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.early_stop = False

        else:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        print("Checkpoint: epoch = %4d, loss = %.4f, best_epoch = %4d, best_loss = %.4f"%(epoch, score, self.best_epoch, self.best_score))

def calcError(model_name:str, outs:np.array, gts:np.array, draw_err=False):

    dir_save = './result_save/' + args['use_dataset'] +'/' + model_name + '/'
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    
    n_sample = gts.shape[0]
    n_predts = gts.shape[1]

    abs_err = np.abs(gts - outs)
    err_x = pd.DataFrame(abs_err[:,:,0]).round(2)
    err_y = pd.DataFrame(abs_err[:,:,1]).round(2)
    err_2 = pd.DataFrame(np.sqrt(err_x**2 + err_y**2)).round(2)

    err_x.to_csv(dir_save+'err_x.csv', mode='w', index=False, header=False)
    err_y.to_csv(dir_save+'err_y.csv', mode='w', index=False, header=False)
    err_2.to_csv(dir_save+'err_2.csv', mode='w', index=False, header=False)

    pred_ADE = np.mean(err_2.values)
    pred_FDE = np.mean(err_2.values[:,-1])
    pred_MR_2 = np.sum(err_2.values[:,-1] > 2.0) / n_sample
    pred_MR_5 = np.sum(err_2.values[:,-1] > 5.0) / n_sample
    print("ADE=%.2f, FDE=%.2f, MR_2=%.2f, MR_5=%.2f"%(pred_ADE, pred_FDE, pred_MR_2*100, pred_MR_5*100))

    ts = [1.0, 2.0, 3.0, 4.0, 5.0]
    idxs = [int(t_*args['fps'])-1 for t_ in ts]
    cols = [str(t_) for t_ in ts]
    ex = pd.DataFrame(err_x.values[:,idxs], columns=cols)
    ey = pd.DataFrame(err_y.values[:,idxs], columns=cols)
    e2 = pd.DataFrame(err_2.values[:,idxs], columns=cols)
    
    # 计算 x 误差指标
    ex_info = pd.DataFrame(columns=cols)
    ex_info.loc['mean'] = ex.mean()
    ex_info.loc['std'] = ex.std()
    ex_info.loc['rmse'] = np.sqrt((ex**2).mean())
    ex_info.loc['25%'] = ex.quantile(0.25)
    ex_info.loc['50%'] = ex.quantile(0.50)
    ex_info.loc['75%'] = ex.quantile(0.75)
    ex_info = ex_info.round(2)

    # print(ex_info)

    # 计算 y 误差指标
    ey_info = pd.DataFrame(columns=cols)
    ey_info.loc['mean'] = ey.mean()
    ey_info.loc['std'] = ey.std()
    ey_info.loc['rmse'] = np.sqrt((ey**2).mean())
    ey_info.loc['25%'] = ey.quantile(0.25)
    ey_info.loc['50%'] = ey.quantile(0.50)
    ey_info.loc['75%'] = ey.quantile(0.75)
    ey_info = ey_info.round(2)

    # print(ey_info)

    # 计算位置误差指标
    e2_info = pd.DataFrame(columns=cols)
    e2_info.loc['mean'] = e2.mean()
    e2_info.loc['std'] = e2.std()
    e2_info.loc['rmse'] = np.sqrt((e2**2).mean())
    e2_info.loc['25%'] = e2.quantile(0.25)
    e2_info.loc['50%'] = e2.quantile(0.50)
    e2_info.loc['75%'] = e2.quantile(0.75)
    e2_info = e2_info.round(2)
    print("RMSE: ", e2_info.loc['rmse'].values)

    if not draw_err:
        return

    plt.figure(figsize=(12,8), dpi=150)

    # 绘制 x 误差
    ax_11 = plt.subplot(2, 3, 1)
    plt.title('Error X [m]')
    ex.boxplot(ax=ax_11, sym='', showmeans=True, patch_artist=True, boxprops={'facecolor': 'lightgray'})
    plt.ylim((0.0, 10.0))
    plt.grid(linestyle="--", alpha=0.5)

    ax_21 = plt.subplot(2, 3, 4)
    tb_21 = plt.table(cellText=np.round(ex_info.values, 2), 
                colLabels=ex_info.columns.values, 
                rowLabels=ex_info.index.values, 
                loc='center', 
                cellLoc='center', 
                rowLoc='center')
    tb_21.scale(1.0, 2.0)
    ax_21.axis('off')

    # 绘制 y 误差
    ax_12 = plt.subplot(2, 3, 2)
    plt.title('Error Y [m]')
    ey.boxplot(ax=ax_12, sym='', showmeans=True, patch_artist=True, boxprops={'facecolor': 'lightgray'})
    plt.ylim((0.0, 3.0))
    plt.grid(linestyle="--", alpha=0.5)

    ax_22 = plt.subplot(2, 3, 5)
    tb_22 = plt.table(cellText=np.round(ey_info.values, 2), 
                colLabels=ey_info.columns.values, 
                rowLabels=ey_info.index.values, 
                loc='center', 
                cellLoc='center', 
                rowLoc='center')
    tb_22.scale(1.0, 2.0)
    ax_22.axis('off')

    # 绘制总误差
    ax_13 = plt.subplot(2, 3, 3)
    plt.title('Error L2-Distance [m]')
    e2.boxplot(ax=ax_13, sym='', showmeans=True, patch_artist=True, boxprops={'facecolor': 'lightgray'})
    plt.ylim((0.0, 10.0))
    plt.grid(linestyle="--", alpha=0.5)

    ax_23 = plt.subplot(2, 3, 6)
    tb_23 = plt.table(cellText=np.round(e2_info.values, 2), 
                colLabels=e2_info.columns.values, 
                rowLabels=e2_info.index.values, 
                loc='center', 
                cellLoc='center', 
                rowLoc='center')
    tb_23.scale(1.0, 2.0)
    ax_23.axis('off')

    plt.show()

    return ex_info, ey_info, e2_info
