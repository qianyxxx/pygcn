import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):   # 初始化
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)    # 构建第一层 GCN
        self.gc2 = GraphConvolution(nhid, nclass)   # 构建第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))    # 第一层，并用relu激活
        x = F.dropout(x, self.dropout, training=self.training)  # 丢弃一部分特征
        x = self.gc2(x, adj)    # 第二层
        return F.log_softmax(x, dim=1)  # softmax激活函数
