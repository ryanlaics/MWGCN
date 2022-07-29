from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import numpy as np


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class layer_block(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(layer_block, self).__init__()
        self.conv_output = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1), padding=(0, int( (k_size-1)/2 ) ) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))
        self.relu = nn.ReLU()
        
        
    def forward(self, input):
        conv_output = self.conv_output(input) # shape (B, D, N, T)

        conv_output1 = self.conv_output1(input)
        
        output = self.output(conv_output1)

        return self.relu( output+conv_output[...,-output.shape[3]:] )

class multi_scale_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, seq_length, layer_num, kernel_set, layer_norm_affline=True):
        super(multi_scale_block, self).__init__()

        self.seq_length = seq_length
        self.layer_num = layer_num
        self.norm = nn.ModuleList()
        self.scale = nn.ModuleList()

        for i in range(self.layer_num):
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))        
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))


        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))
        for i in range(1, self.layer_num):
            self.scale.append(layer_block(c_out, c_out, kernel_set[i]))

        
    def forward(self, input): # input shape: B D N T

        scale = []
        scale_temp = input
        scale_temp = self.start_conv(scale_temp)
        for i in range(self.layer_num):
            scale_temp = self.scale[i](scale_temp)
            scale.append(scale_temp)

        return scale

class gated_fusion(nn.Module):
    def __init__(self, skip_channels, layer_num, ratio=1):
        super(gated_fusion, self).__init__()
        self.dense1 = nn.Linear(in_features=skip_channels*(layer_num+1), out_features=(layer_num+1)*ratio, bias=False)
        self.dense2 = nn.Linear(in_features=(layer_num+1)*ratio, out_features=(layer_num+1), bias=False)


    def forward(self, input1, input2):

        se = torch.mean(input1, dim=2, keepdim=False)
        se = torch.squeeze(se)

        se = torch.relu(self.dense1(se))
        se = torch.sigmoid(self.dense2(se))

        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)

        x = torch.mul(input2, se)
        x = torch.mean(x, dim=1, keepdim=False)
        return x

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):

        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)

        ho = torch.cat(out,dim=1)

        ho = self.mlp(ho)

        return ho

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device, alpha=3):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num
        
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(dim,dim))
            self.lin2.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        

    def forward(self, idx, scale_set):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        adj_set = []

        for i in range(self.layers):
            nodevec1 = torch.tanh(self.alpha*self.lin1[i](nodevec1*scale_set[i]))
            nodevec2 = torch.tanh(self.alpha*self.lin2[i](nodevec2*scale_set[i]))
            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj0 = F.relu(torch.tanh(self.alpha*a))
        
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj0.topk(self.k,1)
            mask.scatter_(1,t1,s1.fill_(1))
            adj = adj0*mask
            adj_set.append(adj)


        return adj_set

