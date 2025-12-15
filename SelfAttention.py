import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention:
  def __init__(self, X):
    self.X = X

  def attention(self):
    atten = torch.matmul(self.X, self.X.T)
    normed_atten = F.softmax(atten, dim=-1)
    return normed_atten
