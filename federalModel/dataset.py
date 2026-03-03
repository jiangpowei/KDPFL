import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        print(f"self.data shape: {self.data.shape}, idx: {idx}, seq_length: {self.seq_length}")
        x = self.data[idx:idx+self.seq_length, :-1]
        y = self.data[idx+self.seq_length, -1]
        return torch.FloatTensor(x), torch.FloatTensor([y])