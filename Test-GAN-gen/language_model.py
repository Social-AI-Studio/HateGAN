import torch
import torch.nn as nn
import numpy as np
import config
import os
import torch.nn.functional as F
from torch.autograd import Variable
import math

opt=config.parse_opt()
class Word_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Word_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        if opt.DATASET=='dt':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_DATA,'glove_embedding.npy')))
        elif opt.DATASET=='founta':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.FOUNTA_DATA,'glove_embedding.npy'))) 
        elif opt.DATASET=='dt_full':       
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy')))
            #glove_weight=torch.from_numpy(np.load('../doubel/total/dictionary/glove_embedding.npy'))
        elif opt.DATASET=='wz':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.WZ_DATA,'glove_embedding.npy')))
        elif opt.DATASET=='total':
            glove_weight=torch.from_numpy(np.load(os.path.join(opt.TOTAL_DATA,'glove_embedding.npy')))
        #glove_weight=torch.from_numpy(np.load('./glove_embedding.npy'))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

        