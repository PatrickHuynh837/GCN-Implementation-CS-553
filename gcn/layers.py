
import torch
import torch.nn as nn
from gcn.layers import *
from gcn.metrics import *


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(GraphConvolution,self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            output += self.bias
        
        return output

class GCN(nn.Module):
    pass
    

    

