class SelfAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.scale = math.sqrt(self.head_dim)

    self.W_Q = nn.Linear(self.d_model, self.head_dim)
    self.W_K = nn.Linear(self.d_model, self.head_dim)
    self.W_V = nn.Linear(self.d_model, self.d_model)
