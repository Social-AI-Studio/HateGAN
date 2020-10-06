import torch
import torch.nn as nn
import config
import numpy as np
import torch.nn.functional as F

from full_rnn import Full_RNN
from language_model import Word_Embedding
from classifier import SimpleClassifier,SingleClassifier

class Deep_Basic(nn.Module):
    def __init__(self,w_emb,rnn,fc,opt):
        super(Deep_Basic,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.fc=fc
        self.rnn=rnn
        
    def forward(self,text):
        batch_size=text.shape[0]
        w_emb=self.w_emb(text)
        capsule,_=self.rnn(w_emb)
        logits=self.fc(capsule[:,-1,:])
        
        return logits 
        
        
def build_baseline(dataset,opt): 
    opt=config.parse_opt()
    
    #w_emb=Word_Embedding(dataset.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT)
    
    w_emb=Word_Embedding(dataset.dictionary.ntoken(),opt.EMB_DIM,opt.EMB_DROPOUT) 
    
    if opt.DATASET=='total' :
        final_dim=3
    elif opt.DATASET=='dt_full':
        final_dim=3
    elif opt.DATASET=='founta':
        final_dim=4
    else:
        final_dim=2
    fc=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    rnn=Full_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    return Deep_Basic(w_emb,rnn,fc,opt)
    
    