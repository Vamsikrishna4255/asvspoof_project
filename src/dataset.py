import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mel = np.load(self.df.iloc[idx]["path"])
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        x = torch.tensor(mel).unsqueeze(0).float()
        y = torch.tensor(self.df.iloc[idx]["label"]).long()
        return x, y
