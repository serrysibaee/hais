import torch
from torch import nn
from torch.nn import functional as F

# inputs (window * dim)
X = torch.rand(100,256)

# Wq, Wk
head_size = 50
W_k = torch.rand(256,head_size)
W_q = torch.rand(256,head_size)
W_v = torch.rand(256,head_size)


# K, Q
K = X @ W_k # (100,256) * (256*head_size) --> (100,head_size)
Q = X @ W_q # (100,256) * (256*head_size) --> (100,head_size)
V = X @ W_v # (100,256) * (256*head_size) --> (100,head_size)
K_Q = Q@K.T # (100,head_size) * (head_size,100) --> (100,100)

# masking
wei = K_Q

# making the ALiBi (needed part)
pos_mask = torch.stack([torch.arange(0,100) for _ in range(100)])
pos_mask = pos_mask.T.float()
# pos_mask = torch.tril(pos_mask)
# for simplicity number of heads here is one (equation ratio=2^(-8/n))
n_heads = 1 
m = 2**(-8/n_heads)
pos_mask = pos_mask.tril(diagonal=-1).float()
pos_mask = -pos_mask*m
# end of it (could be done in the beggining becacuse it is fixed)

tril = torch.tril(torch.ones(100, 100))
# making the masking 
wei = wei.masked_fill(tril == 0, float('-inf'))
# Adding the pos_mask with wei before softmax
wei = F.softmax(wei+pos_mask, dim=-1)
# doing the last bit of attention
I = wei @ V
I.shape