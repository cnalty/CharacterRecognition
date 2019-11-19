import pandas as pd
import torch
import numpy as np

class Minst_Dataset():
    def __init__(self, data_file):
        raw_data = pd.read_csv(data_file)
        print(raw_data.head())
        self.data = np.array(raw_data)
        print(self.data[:50])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][1:]/255
        data = data.reshape(28,-1)
        data = [data]
        print(data)
        data = torch.tensor(data).float()
        label = self.data[index][0]

        return data, label