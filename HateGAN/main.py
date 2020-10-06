import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random
from torch.utils.data import DataLoader

from dataset import Fake_Data, Real_Data
from layers import Target_LSTM, Generator, Discriminator, Rollout, Toxic_Eval
from train import generate_samples, train_real_seqgan
import config

def create_posfile(save_file,data_file,dict_info):
    data=json.load(open(data_file,'r'))
    total=[]
    for tweet in data:
        #print (tweet)
        tokens=dict_info.tokenize(tweet,False)
        #print (tokens)
        if len(tokens)==0:
            tokens=['2']
        total.append(tokens)
    print ('The length of tweets is:',len(total))
    with open(save_file,'w') as fout:
        for t in total:
            string=' '.join([str(s) for s in t])
            fout.write('%s\n' % string)

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    np.random.seed(opt.SEED)
    random.seed(opt.SEED)
    
    generator=Generator(opt.VOC_SIZE,opt.EMB_DIM,opt.NUM_HIDDEN,opt.EMB_DROPOUT,opt.GEN_DROPOUT).cuda()
    discriminator=Discriminator(opt.NUM_CLASSES,opt.VOC_SIZE,opt.EMB_DIM,opt.FILTER_SIZE,opt.NUM_FILTER,opt.DIS_DROPOUT,opt.EMB_DROPOUT).cuda()
    
    print ('Loading data from real data file...')
    train_set=Real_Data(opt.REAL_FILE,opt,mode='gen')
    if opt.POS_FILE:
        create_posfile(opt.POSITIVE_FILE,opt.REAL_FILE,train_set)
    train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    
    """
    details of the toxic model can be referred in the file of toxic
    the configuration of toxic model is set as default
    """
    dict_path='/home/ruicao/NLP/textual/hate-speech-detection/toxic/dictionary/dictionary.pkl'
    toxic_eval= Toxic_Eval(300,dict_path,128,6,0.3,0.5,opt.SENT_LEN).cuda()
    #initialization of parameters in the pre-toxic model
    save_model=torch.load(opt.PRETOXIC)
    pretrained={}
    for para_name in save_model.keys():
        pretrained[para_name]=save_model[para_name].cuda()
    model_state_dict=toxic_eval.state_dict()
    model_state_dict.update(pretrained)
    toxic_eval.load_state_dict(model_state_dict)
    toxic_eval.train(False)
    
    rollout=Rollout(generator,opt,toxic_eval)
    print ('The length of training set is:',len(train_loader))
    
    print ('Start training for Seq GAN...')
    generator.init_embedding(os.path.join(opt.DICT_PATH,opt.DATASET+'_glove_embedding.npy'))
    discriminator.init_embedding(os.path.join(opt.DICT_PATH,opt.DATASET+'_glove_embedding.npy'))
    train_real_seqgan(generator,discriminator,rollout,train_set,opt)
    exit(0)
    
        