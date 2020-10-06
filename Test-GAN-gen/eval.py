import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math

import baseline
import utils
import config
import os
import pickle as pkl
import string
import re
from preprocessing import clean_text
from tqdm import tqdm
import random
import itertools

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
class Data_Set(object):
    def __init__(self):
        self.opt=config.parse_opt()
        self.length=self.opt.LENGTH
        self.init_dict()
        self.entries=self.load_entries()
        self.num_iters=int(math.ceil( len(self.entries) * 1.0 / opt.BATCH_SIZE ))
        self.last_batch=len(self.entries) % opt.BATCH_SIZE
        self.cur_iter=0
    
    def tokenize(self,text):
        '''url_pattern=re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
        emojis_pattern=re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        hash_pattern=re.compile(r'#\w*')
        single_letter_pattern=re.compile(r'(?<![\w\-])\w(?![\w\-])')
        blank_spaces_pattern=re.compile(r'\s{2,}|\t')
        reserved_pattern=re.compile(r'(RT|rt|FAV|fav|VIA|via)')
        mention_pattern=re.compile(r'@\w*')
        CONTRACTION_MAP = {
            "isn't": "is not",
            "aren't": "are not",
            "con't": "cannot",
            "can't've": "cannot have",
            "you'll've": "your will have",
            "you're": "you are",
            "you've": "you have"
        }
        constraction_pattern=re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),flags=re.IGNORECASE|re.DOTALL)
        Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
        urls=re.sub(pattern=url_pattern, repl='', string=text)
        mentions=re.sub(pattern=mention_pattern, repl='', string=urls)
        hashtag=re.sub(pattern=hash_pattern, repl='', string=mentions)
        reserved=re.sub(pattern=reserved_pattern, repl='', string=hashtag)
        reserved=Whitespace.sub(" ", reserved)
        reserved=constraction_pattern.sub(self.expand_match,reserved)
        punct="[{}]+".format(string.punctuation)
        punctuation=re.sub(punct,'',reserved)
        single=re.sub(pattern=single_letter_pattern, repl='', string=punctuation)
        blank=re.sub(pattern=blank_spaces_pattern, repl=' ', string=single)
        blank=blank.lower().split()'''
        #print blank
        blank=clean_text(text)
        return blank
    
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
        data=pkl.load(open(os.path.join(self.opt.SPLIT_DATASET,'covid.pkl'),'rb'),encoding='iso-8859-1') 
        for line in data:
            tweet=line['sent']
            tokens=self.tokenize(tweet)
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
        self.idx2word.append('UNK')
        self.word2idx['UNK']=len(self.idx2word)-1   
        dump_pkl('./dictionary.pkl',[self.word2idx,self.idx2word])
    
    def create_dict(self):
        self.word_count={}
        self.word2idx={}
        self.idx2word=[]
        self.token_sent()
    
    def create_embedding(self):
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
        
            
        np.save('./glove_embedding.npy',weights)
        
    
    def init_dict(self):
        created_dict=load_pkl('./dictionary.pkl')
        self.word2idx=created_dict[0]
        self.idx2word=created_dict[1]
        print ('Creating Dictionary')
        #self.create_dict()
        self.ntoken()
        print ('Creating Glove Embedding')
        #self.glove_weights=self.create_embedding()
        self.glove_weights=np.load('./glove_embedding.npy')
        
        
    def ntoken(self):
        self.ntokens=len(self.word2idx)
        print ('Number of Tokens:',self.ntokens)
        return self.ntokens
    
    def load_entries(self):
        all_data=pkl.load(open(os.path.join(self.opt.SPLIT_DATASET,'covid.pkl'),'rb'),encoding='iso-8859-1') 
        entries=[]
        for info in all_data:
            sent=info['sent']
            key=info['key']
            entry={
                'sent':sent,
                'key':key
            }
            entries.append(entry)
        return entries
    
    def padding_sent(self,tokens,length):
        if len(tokens)<length:
            padding=[self.ntokens]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
            
    
    def next(self):
        batch_info={}
        batch_key=[]
        batch_tweet=[]
        if self.cur_iter>=self.num_iters:
            raise StopIteration
        if self.cur_iter==self.num_iters-1 and self.last_batch>0:
            cur_entry=self.entries[self.cur_iter*self.opt.BATCH_SIZE:self.last_batch]
        else:
            cur_entry=self.entries[self.cur_iter*self.opt.BATCH_SIZE:(1+self.cur_iter)*self.opt.BATCH_SIZE]
        batch_sent=np.zeros([len(cur_entry),self.length],dtype=np.int64)
        for k in range(len(cur_entry)):
            chunck=cur_entry[k]
            sent=chunck['sent']
            batch_tweet.append(sent)
            key=chunck['key']
            batch_key.append(key)
            tokens=self.get_tokens(sent)
            pad_tokens=self.padding_sent(tokens,self.length)
            batch_sent[k,:]=np.array((pad_tokens),dtype=np.int64)
        batch_info['key']=batch_key
        batch_info['tokens']=torch.from_numpy(batch_sent)
        batch_info['sent']=batch_tweet
        self.cur_iter+=1
        return batch_info
    
    def __len__(self):
        return self.num_iters

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    
    total_set=Data_Set()
    
    constructor='build_baseline'
    model=getattr(baseline,constructor)(total_set,opt).cuda()
    
    """
    load pre-trained model
    """
    save_model=torch.load('./model.pth')
    pretrained={}
    for para_name in save_model.keys():
        if para_name=='w_emb.emb.weight' or para_name=='para_emb.emb.weight' or para_name=='fast_emb.emb.weight' or para_name=='senti_emb.emb.weight':
            print ('Ignore..')
            continue
        else:
            pretrained[para_name]=save_model[para_name].cuda()
    model_state_dict=model.state_dict()
    #print (model_state_dict.keys())
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    
    model.w_emb.init_embedding()
    model.train(False)
    
    
    print ('The length of the dataset is:',len(total_set))
    print ('Number of iteration is:',total_set.num_iters)
    logger=utils.Logger('./'+str(opt.SAVE_NUM)+'.txt')
    iters=0
    for batch_info in total_set:
        if iters % 100==0:
            print ('Iteration:',iters,'for generation')
        with torch.no_grad():
            tokens=batch_info['tokens'].long().cuda()
            key=batch_info['key']
            sent=batch_info['sent']
            probs=model(tokens,key)
            pred=torch.max(probs,dim=1)[1].cpu().numpy()
            for i in range(len(sent)):
                t_sent=sent[i]
                t_key=str(key[i])
                t_pred=str(pred[i])
                total_sent=t_key+' '+t_pred+' '+t_sent
                logger.write(total_sent)
        iters+=1       
    exit(0)
    