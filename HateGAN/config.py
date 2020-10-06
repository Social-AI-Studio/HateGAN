import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--GLOVE_PATH',type=str,default='/home/ruicao/trained/embeddings/glove.6B.300d.txt')    
    parser.add_argument('--DATASET',type=str,default='dt')
    parser.add_argument('--POSITIVE_FILE',type=str,default='./data/real.data')#tokens of all tweets
    parser.add_argument('--HATE_FILE',type=str,default='./data/hate.data')#tokens of hate tweets
    parser.add_argument('--NEG_FILE',type=str,default='./data/gen.data')
    parser.add_argument('--EVAL_FILE',type=str,default='./data/eval.data')
    parser.add_argument('--REAL_EVAL_FILE',type=str,default='./data/real_eval')
    parser.add_argument('--REAL_FILE',type=str,default='/home/ruicao/NLP/datasets/hate-speech/split_data/total.json')#it should be modified when the dataset is different
    parser.add_argument('--REAL_HATE_FILE',type=str,default='/home/ruicao/NLP/datasets/hate-speech/split_data/total_hate.json')#store hate tweets only
    parser.add_argument('--RESULT_PATH',type=str,default='./result')
    parser.add_argument('--DICT_PATH',type=str,default='./dictionary')
    parser.add_argument('--PRETOXIC',type=str,default='/home/ruicao/NLP/textual/hate-speech-detection/toxic/model.pth')
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    
    #hyper parameters
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    parser.add_argument('--NUM_HIDDEN',type=int,default=1024)
    parser.add_argument('--GENERATED_NUM',type=int,default=50000)
    parser.add_argument('--SENT_LEN',type=int,default=20)
    parser.add_argument('--VOC_SIZE',type=int,default=8000)
    parser.add_argument('--DIS_ITERS',type=int,default=4)
    parser.add_argument('--GEN_ITERS',type=int,default=1)
    parser.add_argument('--MC_SAMPLES',type=int,default=16)
    parser.add_argument('--NUM_FILTER',type=list,default=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160])
    parser.add_argument('--FILTER_SIZE',type=list,default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
    
    parser.add_argument('--FUSE_ITERS',type=int,default=200)
    parser.add_argument('--UPDATE_PRE_GEN_TIMES',type=int,default=3)#3
    parser.add_argument('--UPDATE_PRE_DIS_TIMES',type=int,default=3)#3
    parser.add_argument('--UPDATE_MIX_DIS_TIMES',type=int,default=3)#3
    parser.add_argument('--PRE_GEN_EPOCHS',type=int,default=200)#120
    parser.add_argument('--PRE_DIS_EPOCHS',type=int,default=7)#5
    parser.add_argument('--EVAL_ITERS',type=int,default=5)
    
    parser.add_argument('--EMB_DROPOUT',type=int,default=0.5)
    parser.add_argument('--GEN_DROPOUT',type=int,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=int,default=0.5)
    parser.add_argument('--DIS_DROPOUT',type=int,default=0.5)
    parser.add_argument('--UPDATE_RATE',type=int,default=0.8)
    parser.add_argument('--DELTA',type=int,default=0.8)
    
    parser.add_argument('--CUDA_DEVICE',type=int,default=0)
    parser.add_argument('--SEED',type=int,default=1111)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--NUM_CLASSES',type=int,default=2)
    
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--POS_FILE',type=bool,default=True)
    parser.add_argument('--TRAIN_RAW',type=bool,default=True)
    parser.add_argument('--MIN_OCC',type=int,default=3)
    
    args=parser.parse_args()
    return args