# unit test for adjacency matrix transform

import torch
import numpy as np
import scipy.sparse as sp

i=torch.LongTensor([[0,1,1],[2,0,2]])
j=i.t()

adj = sp.coo_matrix((np.ones(i.shape[0]), (i[:, 0], i[:, 1])),  # 构建边的邻接矩阵
                        shape=(5,5),
                        dtype=np.float32)
print(adj.A)
print('~~~~~~~~~~~')
print(adj.T > adj)
print('~~~~~~~~~~~')
print(adj.multiply(adj.T > adj))
print('~~~~~~~~~~~')
print(adj.T.multiply(adj.T > adj))
print('~~~~~~~~~~~')
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
print(adj.A)