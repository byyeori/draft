import numpy as np
import torch
from torch.utils.data import Dataset

class UnifiedDataset(Dataset):
    def __init__(self, imfs, raw, seq_len, pred_len, scaler_imf, scaler_raw):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.imfs = scaler_imf.transform(imfs)
        self.raw  = scaler_raw.transform(raw.reshape(-1,1)).reshape(-1)

        T = len(raw)
        hours = np.arange(T) % 24
        days  = np.arange(T) % 7

        self.exog = np.stack([
            np.sin(2*np.pi*hours/24),
            np.cos(2*np.pi*hours/24),
            np.sin(2*np.pi*days/7),
            np.cos(2*np.pi*days/7),
        ], axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.raw) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_imf = self.imfs[idx:idx+self.seq_len]
        x_raw = self.raw[idx:idx+self.seq_len]
        x_ex  = self.exog[idx:idx+self.seq_len]
        y     = self.raw[idx+self.seq_len:idx+self.seq_len+self.pred_len]

        return (
            torch.FloatTensor(x_imf),
            torch.FloatTensor(x_raw).unsqueeze(-1),
            torch.FloatTensor(x_ex),
            torch.FloatTensor(y).unsqueeze(-1)
        )
