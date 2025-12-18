params = {
    'batch_size': None,
    'seq_len': None,
    'vocab_size': None,
    'd_model': None,
    "num_heads": None,
    "max_len": None
}


class SelfAttention(nn.Module):
  def __init__(self, d_model, num_heads, max_len):
    super().__init__()

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    torch.manual_seed(123)

    self.max_len = max_len
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.scale = math.sqrt(self.head_dim)

    torch.manual_seed(123)
    self.W_Q = nn.Linear(self.d_model,self.d_model, bias=False)
    self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
    self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)

  def forward(self, x):
    Q = self.W_Q(x)
    K = self.W_K(x)
    V = self.W_V(x)
    attn_scores = matmul(Q, K.transpose(-2,-1)) / self.scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    context_vector = matmul(attn_weights, V)
    return context_vector

