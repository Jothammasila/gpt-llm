import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len, qkv_bias=False):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1 / math.sqrt(self.head_dim) # More efficient: multiply instead of divide

        # Single projection is also common, but lets keep it simple for clarity
        self.W_Q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.W_O = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Register causal mask once (best practice)
        self.mask = torch.triu(torch.ones(max_len, max_len), diagonal=1)
        self.register_buffer("mask", self.mask)

    def forward(self, x):
        batch, seq_len, d_model = x.shape # batch, seq_len, d_model
        # Project
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        """
        # Q, K, V outputs are in shape (batch, seq_len, d_model)

        # d_model is not a shape — it is the total embedding dimension:
        # (d_model = num_heads * head_dim ) from the definiton (head_dim = d_model // num_heads)

        # When we do:
        # Q.view(batch, seq_len, num_heads, head_dim)
        # we are reshaping the last dimension (d_model) into two dimensions:
        # (num_heads, head_dim)

        # So we are NOT changing the data, only reorganizing it into multiple heads

        # After this, we transpose:
        # Q.transpose(1, 2)

        # This changes:
        # (batch, seq_len, num_heads, head_dim → (batch, num_heads, seq_len, head_dim)

        # This is done so that each head can process the sequence independently
        # during attention computation.
        """

        # Reshape for multi-head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shape: (batch, heads, seq_len, head_dim)

        # Attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        # shape: (batch, heads, seq_len, seq_len)

        # Apply causal mask
        causal_mask = self.mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

        # Normalization (Softmax)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        context = attn_weights @ V  # (batch, heads, seq_len, head_dim)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # Final projection
        out = self.W_O(context)

        return out