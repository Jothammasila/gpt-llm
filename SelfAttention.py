import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleSelfAttention:
    def __init__(self, X):
        self.X = X  # The input matrix, used for queries, keys, and values.

    def attention(self):
        # Step 1: Compute pairwise attention scores (dot product between X and its transpose)
        atten = torch.matmul(self.X, self.X.T)
        
        # Step 2: Apply softmax to normalize the scores (so they sum to 1)
        normed_atten = F.softmax(atten, dim=-1)
        
        # Step 3: Compute the context vectors as the weighted sum of values (X)
        context_vector = torch.matmul(normed_atten, self.X)
        
        # Return both the attention weights and context-aware representation
        return normed_atten, context_vector

