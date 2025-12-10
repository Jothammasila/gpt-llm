from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
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

def data_loader(text, batch_size=2, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  # Initialise tokenizer
  tokenizer = tiktoken.get_encoding("gpt2")

  # Create Dataset
  dataset = GPTDataset(text, tokenizer, max_length, stride)

  #Create dataloader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
  return dataloader
