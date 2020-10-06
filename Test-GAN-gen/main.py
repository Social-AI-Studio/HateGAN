import torch
import torch.nn as nn
from torch.utils.data import Subset,ConcatDataset

from dataset import Base_Op,Wraped_Data
import baseline
from train import train_for_deep
import utils
import config
import os
import pickle as pkl

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    
    #result saving
    if opt.DATASET=='wz':
        logger=utils.Logger(os.path.join(opt.WZ_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='dt_full':
        logger=utils.Logger(os.path.join(opt.OFFENSIVE_FULL_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='founta':
        logger=utils.Logger(os.path.join(opt.FOUNTA_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='total':
        logger=utils.Logger(os.path.join(opt.TOTAL_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
    
    dictionary=Base_Op()
    dictionary.init_dict()
    split_dataset=pkl.load(open(os.path.join(opt.SPLIT_DATASET,opt.DATASET+'.pkl'),'rb'))
    
    constructor='build_baseline'
    #definitions for criteria
    score=0.0
    f1=0.0
    recall=0.0
    precision=0.0
    m_f1=[]
    m_recall=[]
    m_precision=[]
    num_class=0
    for i in range(opt.CROSS_VAL):
        train_set=Wraped_Data(opt,dictionary,split_dataset,i)
        test_set=Wraped_Data(opt,dictionary,split_dataset,i,'test')
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding()
        s,f,p,r,m_f,m_r,m_p=train_for_deep(model,test_set,opt,train_set)
        score+=s
        f1+=f
        precision+=p
        recall+=r
        num_class=len(m_p)
        print (num_class)
        if i==0:
            m_f1.extend(m_f)
            m_recall.extend(m_r)
            m_precision.extend(m_p)
        else:
            m_f1=[a+b for a,b in zip(m_f1,m_f)]
            m_recall=[a+b for a,b in zip(m_recall,m_r)]
            m_precision=[a+b for a,b in zip(m_precision,m_p)]
        logger.write('validation folder %d' %(i+1))
        logger.write('\teval score: %.2f ' % (s))
        logger.write('\teval precision: %.2f ' % (p))
        logger.write('\teval recall: %.2f ' % (r))
        logger.write('\teval f1: %.2f ' % (f))
        #print (num_class,len(m_f1))
        for k in range(num_class):
            logger.write('validation class %d' %(k))
            logger.write('\teval class precision: %.2f ' % (m_p[k]))
            logger.write('\teval class recall: %.2f ' % (m_r[k]))
            logger.write('\teval class f1: %.2f ' % (m_f[k]))  
            
        
    score/=opt.CROSS_VAL
    f1/=opt.CROSS_VAL
    precision/=opt.CROSS_VAL
    recall/=opt.CROSS_VAL
    print (m_f1,m_precision,m_recall)
    m_f11=[t/opt.CROSS_VAL for t in m_f1]
    m_precision1=[t/opt.CROSS_VAL for t in m_precision]
    m_recall1=[t/opt.CROSS_VAL for t in m_recall]
    logger.write('\n final result')
    logger.write('\teval score: %.2f ' % (score))
    logger.write('\teval precision: %.2f ' % (precision))
    logger.write('\teval recall: %.2f ' % (recall))
    logger.write('\teval f1: %.2f ' % (f1))
    for k in range(len(m_f1)):
        logger.write('\tevaluation for class %d ' % (k))
        logger.write('\teval class precision: %.2f ' % (m_precision1[k]))
        logger.write('\teval class recall: %.2f ' % (m_recall1[k]))
        logger.write('\teval class f1: %.2f ' % (m_f11[k]))
    exit(0)
    