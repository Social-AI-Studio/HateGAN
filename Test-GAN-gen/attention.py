
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from fc import FCNet
from classifier import SimpleClassifier

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention,self).__init__()
        self.opt=opt
        self.v_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.q_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.att=FCNet(self.opt.PROJ_DIM,1,self.opt.FC_DROPOUT)
        self.softmax=nn.Softmax()
        
    def forward(self,v,q):
        v_proj=self.v_proj(v)
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        vq_proj=F.relu(v_proj +q_proj)
        proj=torch.squeeze(self.att(vq_proj))
        w_att=torch.unsqueeze(self.softmax(proj),2)
        vatt=v * w_att
        att=torch.sum(vatt,1)
        return att

class Bilinear_Att(nn.Module):
    def __init__(self,in_a,in_b,bilinear_dim,dropout):
        super(Bilinear_Att,self).__init__()
        self.proj_a=SingleClassifier(in_a,bilinear_dim,dropout)
        self.proj_b=SingleClassifier(in_b,bilinear_dim,dropout)
        self.proj=SingleClassifier(bilinear_dim,1,dropout)
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,a,b):
        proj_a=self.proj_a(a)
        proj_b=self.proj_b(b)
        modi_a=proj_a.transpose(1,2).unsqueeze(3)
        modi_b=proj_b.transpose(1,2).unsqueeze(2)
        final=torch.matmul(modi_a,modi_b).transpose(1,2).transpose(2,3)
        final=torch.squeeze(self.proj(final)) #B* N_a * N_b
        modi=final.view(-1,a.size()[1]*b.size()[1]).contiguous()
        norm_weight=self.softmax(modi).view(-1,a.size()[1],b.size()[1]).contiguous()
        return norm_weight
    
class MFB(nn.Module):
    '''this version does not add nolinear layer'''
    def __init__(self,opt):
        super(MFB,self).__init__()    
        self.proj_x=nn.Linear(opt.NUM_HIDDEN,opt.NUM_HIDDEN*3)
        self.proj_y=nn.Linear(opt.NUM_HIDDEN+5,opt.NUM_HIDDEN*3)
        self.dropout=nn.Dropout(opt.FC_DROPOUT)
        self.opt=opt
        self.final_proj=nn.Linear(opt.NUM_HIDDEN,opt.NUM_HIDDEN)
        
        
    def forward(self,x,y):
        batch_size=x.shape[0]
        proj_x=self.dropout(self.proj_x(x))
        proj_y=self.dropout(self.proj_y(y))
        xy=proj_x * proj_y #B,3H
        reshape_xy=xy.view(batch_size,self.opt.NUM_HIDDEN,-1)
        pool_xy=torch.sum(reshape_xy,dim=2)
        final_xy=self.final_proj(pool_xy)
        sqrt_xy=torch.sqrt(F.relu(final_xy))-torch.sqrt(F.relu(-final_xy))
        norm_xy=F.normalize(sqrt_xy)
        return norm_xy

class Intra(nn.Module):
    '''this version does not add nolinear layer'''
    def __init__(self,opt):
        super(Intra,self).__init__()
        self.opt=opt
        self.softmax=nn.Softmax(dim=2)
        self.proj=nn.Linear(opt.NUM_HIDDEN * 2,opt.NUM_HIDDEN)
        
        
    '''considering the impact of b'''    
    def forward(self,a,b):
        simi_matrix=torch.bmm(a,b.transpose(1,2))/math.sqrt(self.opt.NUM_HIDDEN)# B*dim_a*dim_b
        norm_weight=self.softmax(simi_matrix)
        up_1=torch.bmm(norm_weight,b) + a
        return up_1


class Gate_Attention(nn.Module):
    def __init__(self,num_hidden_a,num_hidden_b,num_hidden):
        super(Gate_Attention,self).__init__()
        self.hidden=num_hidden
        self.w1=nn.Parameter(torch.Tensor(num_hidden_a,num_hidden))
        self.w2=nn.Parameter(torch.Tensor(num_hidden_b,num_hidden))
        self.bias=nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()
        
    def reset_parameter(self):
        stdv1=1. / math.sqrt(self.hidden)
        stdv2=1. / math.sqrt(self.hidden)
        stdv= (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1,stdv1)
        self.w2.data.uniform_(-stdv2,stdv2)
        self.bias.data.uniform_(-stdv,stdv)
        
    def forward(self,a,b):
        wa=torch.matmul(a,self.w1)
        wb=torch.matmul(b,self.w2)
        gated=wa+wb+self.bias
        gate=torch.sigmoid(gated)
        output=gate * a + (1-gate) * b
        return output