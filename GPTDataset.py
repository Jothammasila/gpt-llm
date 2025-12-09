class GPTDataset(torch.utils.data.Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
      super().__init__()

      self.input_ids = []
      self.target_ids = []

      self.stride = stride
      self.text = text
      self.tokenizer = tokenizer
      self.max_length = max_length
      self.stride = stride

      self.token_ids = self.tokenizer.encode(self.text)
      for i in range(0, len(self.token_ids) - self.max_length, stride):
        input_chunk = self.token_ids[i:i+self.max_length]
        target_chunk = self.token_ids[i+1:i+self.max_length+1]
        self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
        self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
