from torch.utils.data import Dataset
import torch
import numpy as np
import copy


class MyDataSet(Dataset):
    def __init__(self, V, batch):
        self.len = batch
        self.src_data = np.random.randint(1, V, size=(batch, 10))
        #print(self.src_data)
        self.src_data[:, 0] = 1
        self.tgt_data = copy.deepcopy(self.src_data)
        self.src_len = 10
        self.tgt_len = 10

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        return self.src_data[idx, :], self.src_len, self.tgt_data[idx, :], self.tgt_len

