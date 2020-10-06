import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random
import math
import os
import json
import pickle as pkl

from dataset import Fake_Data,Real_Data
from utils import GANLoss ,Logger

def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def generate_samples(model,batch_size,generated_num,sent_len,save_file):
    samples=[]
    for _ in range(int(generated_num / batch_size)):
        sample=model.sample(sent_len,batch_size).cpu().data.numpy().tolist()
        samples.extend(sample)
    #print (len(samples))
    with open(save_file,'w') as f:
        for sample in samples:
            #print (samples,len(samples),'\n')
            string=' '.join([str(s) for s in sample])
            f.write('%s\n' % string)

def decode_seq(tokens,idx2word):
    sent=[]
    for t in tokens:
        if t==0:
            break
        else:
            sent.append(idx2word[t])
    result=' '.join(sent)
    return result
            
def generate_real_samples(model,batch_size,generated_num,sent_len,save_file,save_num,dict_info):
    samples=[]
    for _ in range(int(generated_num / batch_size)):
        sample=model.sample(sent_len,batch_size).cpu().data.numpy().tolist()
        samples.extend(sample)
    total=[]
    for token in samples:
        sent=decode_seq(token,dict_info)
        total.append(sent)
    print (len(total))
    json.dump(total,open((save_file+'new_'+str(save_num)+'.json'),'w'))
    print ('Saving generated data for the new eval interval')

def each_epoch(model,data_loader,criterion,optimizer=None,mode='train',reason='gen'):
    total_loss=0.0
    total_words=0.0
    if reason=='gen' and mode=='train':
        idx2word=pkl.load(open(os.path.join('./dictionary','dt_dictionary.pkl'),'rb'))[1]
    for i, (tokens,labels) in enumerate(data_loader):
        if mode=='train':
            #print (labels.shape)
            tokens=tokens.long().cuda()
            labels=labels.long().cuda()
            labels=labels.contiguous().view(-1)
            pred=model.forward(tokens)
            #print (pred.shape)
            if reason=='gen' and i%80==0:
                pred_tokens=torch.max(pred,dim=1)[1].cpu().data.numpy().tolist()#B,L,V -> B,L
                sent=decode_seq(pred_tokens[:20],idx2word)
                print (labels[:20])
                print (pred_tokens[:20])
                print (sent,'\n')
            loss=criterion(pred,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                tokens=tokens.long().cuda()
                labels=labels.long().cuda()
                labels=labels.contiguous().view(-1)
                pred=model.forward(tokens)
                loss=criterion(pred,labels)
        total_loss+=loss.item()
        total_words+=tokens.size(0) * tokens.size(1)
    return math.exp(total_loss / total_words)    

def pretrain_dis(generator,discriminator,criterion,optimizer,epochs,batch_size,length,num_gen,neg_file,pos_file,update_each,logger):
    for epoch in range(epochs):
        print ('Generating fake data for updating discriminator...')
        generate_samples(generator,batch_size,num_gen,length,neg_file)
        dis_set=Fake_Data(pos_file,length,mode='dis',neg_file=neg_file)
        dis_loader=DataLoader(dis_set,batch_size,shuffle=True,num_workers=1)
        for i in range(update_each):
            loss=each_epoch(discriminator,dis_loader,criterion,optimizer,mode='train',reason='dis')
            print ('Epoch %d and test times %d, the loss is %f',(epoch,i,loss))
            logger.write('\nTraining for PD for epoch %d, times %d' % (epoch,i))
            logger.write('\teval loss: %.2f ' % (loss))
                
def pretrain_real_gen(train_loader,generator,gen_criterion,optimizer,epochs,logger):
    for epoch in range(epochs):
        loss=each_epoch(generator,train_loader,gen_criterion,optimizer,mode='train',reason='gen')
        print ('Pretraining generator for %d epoch:',epoch+1)
        print ('\t loss is: %f',loss)
        logger.write('\nTraining for PE for epoch %d' % (epoch))
        logger.write('\ttrain loss: %.2f ' % (loss))            
            
def train_real_seqgan(generator,discriminator,rollout,train_set,opt):
    logger=Logger(os.path.join(opt.RESULT_PATH,str(opt.SAVE_NUM)+'.txt'))
    log_hyperpara(logger,opt)
    print ('Start pretraining for the generator...')
    gen_criterion=nn.NLLLoss(reduction='sum')
    gen_optimizer=optim.Adam(generator.parameters())
    logger.write('Pretraining Generator...')
    train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    pretrain_real_gen(train_loader,generator,gen_criterion,gen_optimizer,opt.PRE_GEN_EPOCHS,logger)
    
    print ('Start pretraining for the discriminator...')
    logger.write('Pretraining Discriminator...')
    dis_criterion=nn.NLLLoss(reduction='sum')
    dis_optimizer=optim.Adam(discriminator.parameters())
    pretrain_dis(generator,discriminator,dis_criterion,dis_optimizer,opt.PRE_DIS_EPOCHS,opt.BATCH_SIZE,opt.SENT_LEN,opt.GENERATED_NUM,opt.NEG_FILE,opt.POSITIVE_FILE,opt.UPDATE_PRE_DIS_TIMES,logger)
    
    ad_gen_loss=GANLoss()
    print ('Starging the adversarial learning ...')
    voc=pkl.load(open(os.path.join(opt.DICT_PATH,opt.DATASET+'_dictionary.pkl'),'rb'))
    dict_info=voc[1]#idx2word
    word_info=voc[0]#word2idx
    #create_posfile(opt.HATE_FILE,opt.REAL_HATE_FILE,train_set) 
    create_posfile(opt.HATE_FILE,opt.REAL_FILE,train_set) 
    if opt.TRAIN_RAW:
        torch.save(generator.state_dict(),'./model/gen.pth')
        torch.save(discriminator.state_dict(),'./model/dis.pth')
    for batch in range(opt.FUSE_ITERS):
        print ('Batch %d for adversarial training...'%(batch+1))
        for it in range(opt.GEN_ITERS):
            samples=generator.sample(opt.SENT_LEN,opt.BATCH_SIZE)
            padding=torch.ones((opt.BATCH_SIZE,1)).type(torch.LongTensor).cuda()
            #print (samples.shape,padding.shape)
            inputs=torch.cat((padding,samples.data),dim=1)[:,:-1]
            target=samples.data.contiguous().view(-1,)
            reward,dis_reward,toxic_reward=rollout.get_reward(samples,opt.MC_SAMPLES,discriminator,dict_info)
            #print (dis_reward,toxic_reward)
            #reward=torch.exp(reward).contiguous().view(-1,)
            reward=reward.contiguous().view(-1,)
            pred=generator.forward(inputs)
            loss=ad_gen_loss(pred,target,reward)
            logger.write('\nAdversarial training for batch %d iteration %d' % (batch,it))
            logger.write('\tadversarial loss: %.2f ' % (loss))
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()
            
        if batch % opt.EVAL_ITERS==0:
            generate_real_samples(generator,opt.BATCH_SIZE,opt.GENERATED_NUM,opt.SENT_LEN,opt.REAL_EVAL_FILE,int(batch/opt.EVAL_ITERS),dict_info)
            #add implementation forevaluation using pre-trained LSTM
          
        rollout.update_params()
        for it in range(opt.DIS_ITERS):
            logger.write('\nDiscriminator training for batch %d iteration %d' % (batch,it))
            generate_samples(generator,opt.BATCH_SIZE,opt.GENERATED_NUM,opt.SENT_LEN,opt.NEG_FILE)
            dis_set=Fake_Data(opt.POSITIVE_FILE,opt.SENT_LEN,mode='dis',neg_file=opt.NEG_FILE)
            dis_loader=DataLoader(dis_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
            for _ in range(opt.UPDATE_MIX_DIS_TIMES):
                loss=each_epoch(discriminator,dis_loader,dis_criterion,dis_optimizer,mode='train',reason='dis')
                logger.write('\tdiscriminator loss: %.2f ' % (loss))
                                
def create_posfile(save_file,data_file,dict_info):
    data=json.load(open(data_file,'r'))
    total=[]
    for tweet in data:
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