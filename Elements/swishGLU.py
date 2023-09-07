import torch
from torch import nn
from torch.nn import functional as F

def swish(x):
  return x*F.sigmoid(x * 1) # beta = 1 

# Example 
# tns = torch.randint(1,10,(3,3))
# print(swish(tns))

# without bias for simplicity
def swishGLU(x,W,V):
  return swish(x@W) * (x@V)

# Example 
# W = torch.rand(3,3)
# V = torch.rand(3,3)
# x = torch.randint(1,10,(1,3)).float()
# print(swishGLU(x,W,V))

# Building feed forward layer with them
class FFN_new(nn.Module):
  def __init__(self, inp,h1, out):
    super().__init__()
    self.W = nn.Linear(inp,h1)
    self.V = nn.Linear(inp,h1)
    self.W2 = nn.Linear(h1,out)

  def swishGLU(self, x,W,V):
    return swish(W(x)) * (V(x))

  def forward(self,x):
    A = self.swishGLU(x,self.W, self.V)
    B = self.W2(A)

    return B 

model = FFN_new(3,4,5)
tns = torch.rand((1,3))

y = model(tns)

y.shape

    