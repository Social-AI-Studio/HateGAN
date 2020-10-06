import os
import pandas as pd
import re
import json
import pickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import config
import itertools
import random
import string
from nltk.tokenize.treebank import TreebankWordTokenizer
from preprocessing import clean_text

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Base_Op(object):
    def __init__(self):
        self.opt=config.parse_opt()
    
    def tokenize(self,x):
        x = clean_text(x).lower().split()
        #print (x)
        return x
    
    def get_tokens(self,sent):
        tokens=self.tokenize(sent)
        #print tokens
        token_num=[]
        for t in tokens:
            if t in self.word2idx:
                token_num.append(self.word2idx[t])
            else:
                token_num.append(self.word2idx['UNK'])
        return token_num
    
    def token_sent(self):
        cur=0
        data=pkl.load(open(os.path.join(self.opt.SPLIT_DATASET,self.opt.DATASET+'.pkl'),'rb'))  
        for j,line in enumerate(data.keys()):
            cur_info=data[line]#it's a list
            for info in cur_info:
                tweet=info['sent']
                tokens=self.tokenize(tweet)
                for t in tokens:
                    if t not in self.word_count:
                        self.word_count[t]=1
                    else:
                        self.word_count[t]+=1
        
        if self.opt.ADD_GEN==True:
            hate_data=open('../generated_tweets.txt','r').readlines()
            for line in hate_data:
                tokens=self.tokenize(line.strip())
                for t in tokens:
                    if t not in self.word_count:
                        self.word_count[t]=1
                    else:
                        self.word_count[t]+=1
        for word in self.word_count.keys():
            if self.word_count[word]>=self.opt.MIN_OCC:
                self.word2idx[word]=cur
                self.idx2word.append(word)
                cur+=1
        if 'UNK' not in self.word2idx:
            self.idx2word.append('UNK')
            self.word2idx['UNK']=len(self.idx2word)-1   
        if self.opt.DATASET=='dt':
            dump_pkl(os.path.join(self.opt.OFFENSIVE_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='founta':
            dump_pkl(os.path.join(self.opt.FOUNTA_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='dt_full':
            dump_pkl(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='wz':
            dump_pkl(os.path.join(self.opt.WZ_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
        elif self.opt.DATASET=='total':
            dump_pkl(os.path.join(self.opt.TOTAL_DATA,'dictionary.pkl'),[self.word2idx,self.idx2word])
    
    def create_dict(self):
        self.word_count={}
        self.word2idx={}
        self.idx2word=[]
        self.token_sent()
    
    def create_embedding(self):
        print (self.opt.DATASET)
        word2emb={}
        with open(self.opt.GLOVE_PATH,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[0].split(' '))-1
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for entry in entries:
            word=entry.split(' ')[0]
            word2emb[word]=np.array(list(map(float,entry.split(' ')[1:])))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
            
        if self.opt.DATASET=='dt':
            np.save(os.path.join(self.opt.OFFENSIVE_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='founta':
            np.save(os.path.join(self.opt.FOUNTA_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='dt_full':
            np.save(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='wz':
            np.save(os.path.join(self.opt.WZ_DATA,'glove_embedding.npy'),weights)
        elif self.opt.DATASET=='total':
            np.save(os.path.join(self.opt.TOTAL_DATA,'glove_embedding.npy'),weights)
        return weights
    
    def init_dict(self):
        if self.opt.CREATE_DICT:
            print ('Creating Dictionary...')
            self.create_dict()
        else:
            print ('Loading Dictionary...')
            if self.opt.DATASET=='dt':
                created_dict=load_pkl(os.path.join(self.opt.OFFENSIVE_DATA,'dictionary.pkl'))
            elif self.opt.DATASET=='founta':
                created_dict=pkl.load(open(os.path.join(self.opt.FOUNTA_DATA,'dictionary.pkl'),'rb'),encoding='iso-8859-1') 
            elif self.opt.DATASET=='dt_full':
                created_dict=load_pkl(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'dictionary.pkl'))
                #created_dict=load_pkl('../doubel/total/dictionary/dictionary.pkl')
            elif self.opt.DATASET=='wz':
                created_dict=load_pkl(os.path.join(self.opt.WZ_DATA,'dictionary.pkl'))
            elif self.opt.DATASET=='total':
                created_dict=load_pkl(os.path.join(self.opt.TOTAL_DATA,'dictionary.pkl'))
                
            
            self.word2idx=created_dict[0]
            self.idx2word=created_dict[1]
            
        if self.opt.CREATE_EMB:
            print ('Creating Embedding...;')
            self.glove_weights=self.create_embedding()
        else:
            print ('Loading Embedding...')
            if self.opt.DATASET=='dt':
                self.glove_weights=np.load(os.path.join(self.opt.OFFENSIVE_DATA,'glove_embedding.npy'))
            elif self.opt.DATASET=='founta':
                self.glove_weights=np.load(os.path.join(self.opt.FOUNTA_DATA,'glove_embedding.npy'))
            elif self.opt.DATASET=='dt_full':
                self.glove_weights=np.load(os.path.join(self.opt.OFFENSIVE_FULL_DATA,'glove_embedding.npy'))
            elif self.opt.DATASET=='wz':
                self.glove_weights=np.load(os.path.join(self.opt.WZ_DATA,'glove_embedding.npy'))
            elif self.opt.DATASET=='total':
                self.glove_weights=np.load(os.path.join(self.opt.TOTAL_DATA,'glove_embedding.npy'))
        self.ntoken()
        
    def ntoken(self):
        self.ntokens=len(self.word2idx)
        print ('Number of Tokens:',self.ntokens)
        return self.ntokens
    
    
    def __len__(self):
        return len(self.word2idx)
        
class Wraped_Data(Base_Op):
    def __init__(self,opt,dictionary,split_data,test_num,mode='training'):
        super(Wraped_Data,self).__init__()
        self.opt=config.parse_opt()
        self.dictionary=dictionary
        self.split_data=split_data
        self.test_num=test_num
        self.mode=mode
        if self.opt.ADD_GEN:
            self.gen_data=open('../generated_tweets.txt','r').readlines()
            random.shuffle(self.gen_data)
            self.gen_data=self.gen_data[:self.opt.NUM_ADD]
            print ('The length of generated hate speech is:',len(self.gen_data))
        self.entries=self.load_tr_val_entries()
        if self.opt.DATASET=='dt' or self.opt.DATASET=='wz'  :
            self.classes=2
        elif self.opt.DATASET=='dt_full':
            self.classes=3
        elif self.opt.DATASET=='founta':
            self.classes=4
        self.tokenize()
        self.tensorize()
   
    def load_tr_val_entries(self):
        all_data=[]
        #loading dataset for training and testing
        if self.mode=='training':
            for i in range(self.opt.CROSS_VAL):
                if i==self.test_num:
                    continue
                all_data.extend(self.split_data[str(i)])
        else:
            all_data.extend(self.split_data[str(self.test_num)])
        #classify types of tweet
        entries=[]
        count=0
        for info in all_data:
            sent=info['sent']
            if self.opt.DATASET=='total':
                label=info['answer']
                #if label==0:
                #   print (sent)
            else:
                label=info['label']
            if self.opt.DATASET=='wz':
                if label==0 or label==1:
                    label=0
                else:
                    label=1
            entry={
                'tweet':sent,
                'answer':label
            }
            entries.append(entry)
        if self.opt.ADD_GEN and self.mode=='training':
            for line in self.gen_data:
                entry={
                    'tweet':line.strip(),
                    'answer':0
                }
                entries.append(entry)
                count+=1
            print ('The length of generated hate speech is:',count)
        return entries
    
    def padding_sent(self,tokens,length):
        if len(tokens)<length:
            padding=[self.dictionary.ntokens]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
   
    def tokenize(self):
        print('Tokenize Tweets...')
        length=self.opt.LENGTH
        for entry in tqdm(self.entries):
            tokens=self.dictionary.get_tokens(entry['tweet'])
            pad_tokens=self.padding_sent(tokens,length)
            entry['tokens']=np.array((pad_tokens),dtype=np.int64)
            
    def tensorize(self):
        print ('Tesnsorize all Information...')
        count=0
        for entry in tqdm(self.entries):
            entry['text_tokens']=torch.from_numpy(entry['tokens'])
            target=torch.from_numpy(np.zeros((self.classes),dtype=np.float32))
            target[entry['answer']]=1.0
            entry['label']=target
                       
    def __getitem__(self,index):
        entry=self.entries[index]
        tweet=entry['text_tokens']
        label=entry['label']
        return tweet,label
        
        
    def __len__(self):
        return len(self.entries)
    
