import os
import random
import pickle as pkl

import numpy as np
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

class Toxic_Eval(nn.Module):
    #toxic, threat, and target close to one
    def __init__(self,emb_dim,dict_path,num_hidden,final_dim,emb_dropout,fc_dropout,sent_len):
        super(Toxic_Eval,self).__init__()
        self.word2idx=pkl.load(open(dict_path,'rb'))[0]
        self.vocab_size=len(self.word2idx)
        self.emb=nn.Embedding(self.vocab_size+1,emb_dim,padding_idx=self.vocab_size)
        self.lstm1=nn.LSTM(emb_dim,num_hidden,bidirectional=True,batch_first=True)
        self.lstm2=nn.LSTM(num_hidden*2,num_hidden,bidirectional=True,batch_first=True)
        self.out=nn.Linear(num_hidden*4,6)
        
        self.emb_drop=nn.Dropout(emb_dropout)
        self.length=sent_len
        self.crit=nn.BCELoss(reduce=False)
        
    def get_tokens(self,s):
        tokens=[]
        for w in s:
            if w in self.word2idx:
                tokens.append(self.word2idx[w])
            else:
                tokens.append(self.word2idx['UNK'])
        return tokens
    
    def pad_sent(self,tokens,length):
        if len(tokens)<length:
            padding=[self.vocab_size]*(llength-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
    
    def tokenize(self,sent):
        #sentences to tokens and then padding and tensorize
        #the length of the sentence is set as 20
        batch_size=len(sent)
        length=len(sent[1])
        tensored_sent=np.zeros([batch_size,length],dtype=np.int64)
        for i in range(batch_size):
            s=sent[i]
            tokens=self.pad_sent(self.get_tokens(s),length)
            tensored_sent[i,:]=np.array((tokens),dtype=np.int64)
        tensored_sent=torch.from_numpy(tensored_sent).cuda()
        return tensored_sent
    
    def decode_seq(self,tokens,dict_info):
        #print (tokens.shape)
        tokens=tokens.cpu().data.numpy().tolist()
        #print (len(tokens))
        sents=[]
        for t in tokens:
            s=[]
            for w in t:
                if t==0:
                    break
                else:
                    s.append(dict_info[w])
            sents.append(s)
        #print (len(sents))
        return sents
    
    def get_loss(self,pred):
        batch_size=pred.shape[0]
        #print(batch_size)
        label=torch.ones(batch_size,3).float().cuda()
        indices=torch.tensor([0,5]).cuda()
        result=torch.index_select(pred,1,indices)
        #loss=self.crit(result,label)
        return result
    
    def forward(self,pre_tokens,dict_info):
        sent=self.decode_seq(pre_tokens,dict_info)
        token_tensor=self.tokenize(sent)
        t_emb=self.emb(token_tensor)
        t_emb=self.emb_drop(t_emb)
        hidden1,_=self.lstm1(t_emb)
        hidden2,_=self.lstm2(hidden1)
        avg_pool=torch.mean(hidden2,1)
        max_pool,_=torch.max(hidden2,1)
        concat_hidden=torch.cat((max_pool,avg_pool),1)
        result=torch.sigmoid(self.out(concat_hidden)) 
        final=self.get_loss(result)
        return final

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, opt, toxic):
        self.ori_model = model
        self.own_model=copy.deepcopy(model)
        '''model_state_dict=self.own_model.state_dict()
        pretrained={}
        for para_name in model_state_dict.keys():
            pretrained[para_name]=model.state_dict()[para_name].cuda()
        model_state_dict.update(pretrained)
        self.own_model.load_state_dict(model_state_dict)'''
        self.update_rate = opt.UPDATE_RATE
        self.toxic=toxic
        self.opt=opt

    
        
    def get_reward(self, x, num, discriminator,dict_info):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                #print (i,l)
                state= x[:, 0:l]
                #print (state.shape)
                samples = self.own_model.sample(seq_len, batch_size, state=state)
                pred = discriminator(samples)
                dis_cri=torch.exp(pred[:,1])
                #dis_cri=pred[:,1]
                #print (dis_cri)
                with torch.no_grad():
                    toxic_loss= torch.sum(self.toxic(samples,dict_info),dim=1)/2
                #toxic_loss= self.toxic(samples,dict_info)
                #print (toxic_loss)
                total_pred = (self.opt.DELTA*dis_cri+toxic_loss).unsqueeze(1)
                #total_pred=dis_cri
                if i == 0:
                    rewards.append(total_pred)
                else:
                    rewards[l-1] += total_pred

            # for the last token
            pred = discriminator(x)
            dis_cri=torch.exp(pred[:,1])
            #pred = pred.data[:, 1].unsqueeze(1)
            with torch.no_grad():
                toxic_loss= torch.sum(self.toxic(x,dict_info),dim=1)/2
            total_pred = (self.opt.DELTA*dis_cri+toxic_loss)
            total_pred=total_pred.unsqueeze(1)
            #print (total_pred.shape)
            #total_pred=dis_cri
            if i == 0:
                rewards.append(total_pred)
            else:
                rewards[seq_len-1] += total_pred
        rewards=torch.cat(rewards,dim=1)
        rewards = rewards / (1.0 * num) # batch_size * seq_len
        return rewards,dis_cri,toxic_loss

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

class FC_Net(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(FC_Net,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.relu=nn.ReLU()
        self.linear=weight_norm(nn.Linear(in_dim,out_dim),dim=None)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        logits=self.dropout(self.linear(x))
        return logits

class SimpleClassifier(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout):
        super(SimpleClassifier,self).__init__()
        layer=[
            weight_norm(nn.Linear(in_dim,hid_dim),dim=None),
            nn.ReLU(),
            nn.Dropout(dropout,inplace=True),
            weight_norm(nn.Linear(hid_dim,out_dim),dim=None)
        ]
        self.main=nn.Sequential(*layer)
        
    def forward(self,x):
        logits=self.main(x)
        return logits
    
class Word_Embedding(nn.Module):
    def __init__(self,emb_dim,ntoken,dropout,emb_dir=None):
        super(Word_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim
        self.emb_dir=emb_dir
        if emb_dir is not None:
            self.init_embedding()

    def init_embedding(self):
        print ('Initializing glove Embedding...')
        glove_weight=torch.from_numpy(np.load(self.emb_dir))
        self.emb.weight.data[:self.ntoken]=glove_weight
        
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

class Target_LSTM(nn.Module):
    """Target Lstm """
    def __init__(self, final_dim, emb_dim, hidden_dim,emb_dropout,fc_dropout):
        super(Target_LSTM, self).__init__()
        self.final_dim=final_dim
        #self.emb=Word_Embedding(emb_dim,final_dim,emb_dropout)
        self.emb=nn.Embedding(final_dim,emb_dim)
        self.lstm=nn.LSTM(emb_dim,hidden_dim,batch_first=True)
        self.classifier=FC_Net(hidden_dim,final_dim,fc_dropout)
        self.hidden_dim=hidden_dim
        self.init_params()
        
    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0,1)

    def init_hidden(self,batch_size):
        h=Variable(torch.zeros(1,batch_size,self.hidden_dim).cuda())
        c=Variable(torch.zeros(1,batch_size,self.hidden_dim).cuda())
        return h,c
     
    def step(self,x,h,c):
        emb_x=self.emb(x)
        output,(h,c)=self.lstm(emb_x,(h,c))
        pred=F.softmax(self.classifier(output.contiguous().view(-1,self.hidden_dim)),dim=1)
        return pred,h,c
        
    def sample(self,seq_len,batch_size):
        with torch.no_grad():
            x=Variable(torch.ones(batch_size,1).long().cuda())
            h,c=self.init_hidden(batch_size)
            samples=[]
            for i in range(seq_len):
                output,h,c=self.step(x,h,c)
                x=output.multinomial(1)
                samples.append(x)
        output=torch.cat(samples,dim=1)
        return output
        
    def forward(self, x):
        emb_x=self.emb(x)
        h0,c0=self.init_hidden(x.size(0))
        output,(h,c)=self.lstm(emb_x,(h0,c0))
        logits=self.classifier(output.contiguous().view(-1,self.hidden_dim))
        pred=F.log_softmax(logits,dim=1)
        return pred
    
class Generator(nn.Module):
    """Target Lstm """
    def __init__(self, final_dim, emb_dim, hidden_dim,emb_dropout,fc_dropout):
        super(Generator, self).__init__()
        self.final_dim=final_dim
        '''self.emb=Word_Embedding(emb_dim,final_dim,emb_dropout)
        self.lstm=nn.LSTM(emb_dim,hidden_dim,batch_first=True)
        self.classifier=FC_Net(hidden_dim,final_dim,fc_dropout)'''
        self.emb=nn.Embedding(final_dim,emb_dim)
        self.lstm=nn.LSTM(emb_dim,hidden_dim,batch_first=True)
        self.classifier=nn.Linear(hidden_dim,final_dim)
        self.hidden_dim=hidden_dim
        #print ('Initializing hidden states for generator...')
        self.init_params()
        
    def init_params(self):
        print ('Initializing parameters for Generator..')
        for param in self.parameters():
            param.data.normal_(-0.05,0.05)
    
    def init_embedding(self,path):
        glove_weight=torch.from_numpy(np.load(path))
        self.emb.weight.data=glove_weight.cuda()
    
    def init_hidden(self,batch_size):
        h=Variable(torch.zeros(1,batch_size,self.hidden_dim)).cuda()
        c=Variable(torch.zeros(1,batch_size,self.hidden_dim)).cuda()
        return h,c
    
    def step(self,x,h,c):
        emb_x=self.emb(x)
        output,(h,c)=self.lstm(emb_x,(h,c))
        pred=F.softmax(self.classifier(output.contiguous().view(-1,self.hidden_dim)),dim=1)
        return pred,h,c
        
    def sample(self,seq_len,batch_size,state=None):
        samples=[]
        #print (state)
        x=Variable(torch.ones(batch_size,1).long().cuda())
        h,c=self.init_hidden(batch_size)
        if state is None:
            flag=True
        else:
            flag=False
        if flag:
            for i in range(seq_len):
                output,h,c=self.step(x,h,c)
                x=output.multinomial(1)
                samples.append(x)
        else:
            given_len=state.size(1)
            state_list=state.chunk(state.size(1),dim=1)
            for i in range(given_len):
                #print (state_list[i].shape)
                output,h,c=self.step(state_list[i],h,c)
                samples.append(state_list[i])
            x=output.multinomial(1)#B,1
            for i in range(given_len,seq_len):
                samples.append(x)
                output,h,c=self.step(x,h,c)
                x=output.multinomial(1)
        output=torch.cat(samples,dim=1)
        return output
        
    def forward(self, x):
        emb_x=self.emb(x)
        h0,c0=self.init_hidden(x.size(0))
        output,(h,c)=self.lstm(emb_x,(h0,c0))
        logits=self.classifier(output.contiguous().view(-1,self.hidden_dim))
        pred=F.log_softmax(logits,dim=1)
        return pred
    
class Discriminator(nn.Module):

    def __init__(self, num_classes, final_dim, emb_dim, filter_sizes, num_filters, fc_dropout,emb_dropout):
        super(Discriminator, self).__init__()
        #self.emb = Word_Embedding(emb_dim, final_dim, emb_dropout)
        self.emb=nn.Embedding(final_dim,emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.lin = FC_Net(sum(num_filters), num_classes, fc_dropout)
        self.init_parameters()

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax((self.lin(pred)),dim=1)
        return pred

    def init_embedding(self,path):
        glove_weight=torch.from_numpy(np.load(path))
        self.emb.weight.data=glove_weight.cuda()
    
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)