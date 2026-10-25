import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .ernie_rna.tasks.ernie_rna import *
from .ernie_rna.models.ernie_rna import *
from .ernie_rna.criterions.ernie_rna import *


def prepare_input_for_ernierna(index, seq_len):
    
    shorten_index = index[:seq_len+2]
    one_d = torch.from_numpy(shorten_index).long().reshape(1,-1)
    two_d = np.zeros((1,seq_len+2,seq_len+2))
    two_d[0,:,:] = creatmat(shorten_index.astype(int),base_range=1,lamda=0.8)
    two_d = two_d.transpose(1,2,0)
    two_d = torch.from_numpy(two_d).reshape(1,seq_len+2,seq_len+2,1)
    
    return one_d, two_d




def load_pretrained_ernierna(mlm_pretrained_model_path,arg_overrides):
    rna_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),arg_overrides=arg_overrides)
    model_pretrained = rna_models[0]
    return model_pretrained

def gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y,lamda=0.8):
    if x == 5 and y == 6:
        return 2
    elif x == 4 and y == 7:
        return 3
    elif x == 4 and y == 6:
        return lamda
    elif x == 6 and y == 5:
        return 2
    elif x == 7 and y == 4:
        return 3
    elif x == 6 and y == 4:
        return lamda
    else:
        return 0
    
base_range_lst = [1]
lamda_lst = [0.8]

def creatmat(data, base_range=30, lamda=0.8):
    paird_map = np.array([[paired(i,j,lamda) for i in range(30)] for j in range(30)])
    data_index = np.arange(0,len(data))
    # np.indices((2,2))    
    coefficient = np.zeros([len(data),len(data)])
    # mat = np.zeros((len(data),len(data))) 
    score_mask = np.full((len(data),len(data)),True)
    for add in range(base_range):
        data_index_x = data_index - add
        data_index_y = data_index + add
        score_mask = ((data_index_x >= 0)[:,None] & (data_index_y < len(data))[None,:]) & score_mask
        data_index_x,data_index_y = np.meshgrid(data_index_x.clip(0,len(data) - 1),data_index_y.clip(0,len(data) - 1),indexing='ij')
        score = paird_map[data[data_index_x],data[data_index_y]]
        score_mask = score_mask & (score != 0)
        
        coefficient = coefficient + score * score_mask * gaussian(add)
        if ~(score_mask.any()) :
            break
    score_mask = coefficient > 0
    for add in range(1,base_range):
        data_index_x = data_index + add
        data_index_y = data_index - add
        score_mask = ((data_index_x < len(data))[:,None] & (data_index_y >= 0)[None,:]) & score_mask
        data_index_x,data_index_y = np.meshgrid(data_index_x.clip(0,len(data) - 1),data_index_y.clip(0,len(data) - 1),indexing='ij')
        score = paird_map[data[data_index_x],data[data_index_y]]
        score_mask = score_mask & (score != 0)
        coefficient = coefficient + score * score_mask * gaussian(add)
        if ~(score_mask.any()) :
            break
    return coefficient




    

class ErnieRNAOnestage(nn.Module):
    '''
    input_x: shape like: [B,L+2], one for "cls" token, another for "eos" token
    input_twod_input: shape like: [1,L+2,L+2,1], calculated from input_x
    outpus: shape like: [B,L+2,768], 768 is the dim of embedding extracted by pre-train model
    '''
    def __init__(self, sentence_encoder):
        super().__init__() 
        self.sentence_encoder = sentence_encoder
    def forward(self,x,twod_input,return_attn_map = False,i=12,j=5,layer_idx=12):
        _,attn_map_lst,out_dict = self.sentence_encoder(x,twod_tokens=twod_input,is_twod=True,extra_only=True, masked_only=False)
        x = torch.stack(out_dict['inner_states'][1:]).transpose(1,2) # (12,T,B,C) -> (12,B,T,C)
        if layer_idx != 12:
            x = x[layer_idx,:,:,:].unsqueeze(0)

        if return_attn_map:
            L = attn_map_lst[0].shape[2]
            attnmap = torch.stack(attn_map_lst).transpose(0,1) # (13,B,12,T,T) -> (B,13,12,T,T)
            # atten1 = F.softmax(attn_map_lst[i][0,j], dim=-1)
            if i == 13 and j == 12:
                attnmap = attnmap.view(156,L,L)
                return attnmap
            elif i == 13:
                return attnmap[0,:,j,:,:]
            elif j == 12:
                return attnmap[0,i,:,:,:]
            else:
                return attnmap[0,i,j,:,:]
        return x
