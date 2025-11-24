import torch
import torch.nn as nn
from .base_model import BaseModel
from config import SEQ_LEN, PRED_LEN


class TFTModel(BaseModel):
    def __init__(self, hidden=64):
        super().__init__()

        self.vs = nn.Linear(6, hidden)  # raw + exog(4) + IMF summary

        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)

        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)

        self.proj = nn.Linear(SEQ_LEN*hidden, PRED_LEN)

    def forward(self, x_imf, x_raw, x_ex):
        imf_summary = x_imf.mean(dim=-1, keepdim=True)
        x = torch.cat([x_raw, x_ex, imf_summary], dim=-1)
        x = self.vs(x)

        out, _ = self.lstm(x)

        attn, _ = self.attn(out, out, out)
        out = out + attn

        out = out.reshape(out.size(0), -1)
        return self.proj(out).unsqueeze(-1)
