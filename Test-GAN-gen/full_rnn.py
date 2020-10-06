import torch
import torch.nn as nn
import numpy as np
import config
from torch.autograd import Variable
import torch.nn.functional as F

class Gate_combine_three(nn.Module):
    def __init__(self,hidden,mid,dropout):
        super(Gate_combine_three,self).__init__()
        self.opt=config.parse_opt()
        self.f_proj=nn.Linear(hidden,mid)
        self.a_proj=nn.Linear(hidden,mid)
        self.f_att=nn.Linear(mid,1)
        self.a_att=nn.Linear(mid,1)
        self.sig=nn.Sigmoid()
        self.dropout=nn.Dropout(dropout)


    def forward(self,f,a,q):
        
        f_proj=self.f_proj(f+q)
        f_proj=self.dropout(f_proj)
        f_g = self.sig(self.f_att(f_proj))
        
        a_proj=self.a_proj(a+q)
        a_proj=self.dropout(a_proj)
        a_g = self.sig(self.a_att(a_proj))
        
        fa_comb=f_g*f+a_g*a+q
        
        return fa_comb    
    
class Full_RNN(nn.Module):
    def __init__(self,in_dim,num_hidden,num_layer,bidirect,dropout,rnn_type='LSTM'):
        super(Full_RNN,self).__init__()
        rnn_cls=nn.LSTM if rnn_type=='LSTM' else nn.GRU
        self.rnn=rnn_cls(in_dim,num_hidden,num_layer,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.num_layer=num_layer
        self.rnn_type=rnn_type
        self.num_bidirect=1+int(bidirect)
        
    def init_hidden(self,batch):
        weight=next(self.parameters()).data
        hid_shape=(self.num_layer * self.num_bidirect,batch,self.num_hidden)
        if self.rnn_type =='LSTM':
            return (Variable(weight.new(*hid_shape).zero_().cuda()),
                    Variable(weight.new(*hid_shape).zero_().cuda()))
        else:
            return Variable(weight.new(*hid_shape).zero_()).cuda()
    
    def forward(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        hidden=hidden[0].squeeze()
        return output,hidden

class Capsule(nn.Module):
    def __init__(self,opt):
        super(Capsule,self).__init__()
        self.opt=opt
        
    def squash(self,vec):
        norm_vec=torch.sum(vec*vec,dim=1,keepdim=True)
        output_vec=norm_vec * vec / ((1. + torch.sqrt(norm_vec)) * torch.sqrt(norm_vec))
        return output_vec
    
    def forward(self,input_matrix):
        batch_size=input_matrix.shape[0]#B,num_routine,hidden
        b_ij=torch.zeros(batch_size,self.opt.NUM_ROUTINE,1).cuda()
        '''requirement: the dimesion of input vectors should be the same'''
        #print input_matrix.shape,b_ij.shape
        for iteration in range(self.opt.NUM_ITER):
            if iteration>0:
                b_ij=b_ij + input_matrix * final_matrix
            c_ij=F.softmax(b_ij,dim=1)   
            final_matrix= c_ij * input_matrix #B,num_routine,vec_dim
            final_vec=torch.sum(final_matrix,1)
            final_vec=self.squash(final_vec)
            final_matrix=final_vec.repeat(1,self.opt.NUM_ROUTINE).view(-1,self.opt.NUM_ROUTINE,self.opt.NUM_HIDDEN)
            #print final_matrix.shape
        return final_vec
    
class CNN_Model(nn.Module):
    def __init__(self,in_dim,num_hidden,opt):
        super(CNN_Model,self).__init__()
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.dropout=opt.CNN_DROPOUT
        filter_sizes=[int(fsz) for fsz in opt.FILTER_SIZE.split(',')]
        self.conv=nn.ModuleList([nn.Conv2d(1,opt.NUM_FILTER,(fsz,in_dim)) for fsz in filter_sizes])
        
    def forward(self,emb):
        emb=emb.unsqueeze(1)
        conv_result=[F.relu(conv(emb)) for conv in self.conv]
        mid=[torch.squeeze(x_i).transpose(1,2).contiguous() for x_i in conv_result]
        
        return mid
    
class Part_RNN(nn.Module):
    def __init__(self,in_dim,num_hidden,num_layer,bidirect,dropout,rnn_type='LSTM'):
        super(Part_RNN,self).__init__()
        rnn_cls=nn.LSTM if rnn_type=='LSTM' else nn.GRU
        self.rnn=rnn_cls(in_dim,num_hidden,num_layer,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.num_layer=num_layer
        self.rnn_type=rnn_type
        self.num_bidirect=1+int(bidirect)
        
    def init_hidden(self,batch):
        weight=next(self.parameters()).data
        hid_shape=(self.num_layer * self.num_bidirect,batch,self.num_hidden)
        if self.rnn_type =='LSTM':
            return (Variable(weight.new(*hid_shape).zero_().cuda()),
                    Variable(weight.new(*hid_shape).zero_().cuda()))
        else:
            return Variable(weight.new(*hid_shape).zero_()).cuda()
    
    def forward(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        return output[:,-1,:]