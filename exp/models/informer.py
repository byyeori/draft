import torch
import torch.nn as nn
from .base_model import BaseModel
from config import SEQ_LEN, PRED_LEN


class InformerModel(BaseModel):
    def __init__(self, d_model=64, heads=4):
        super().__init__()
        self.enc_in = nn.Linear(6, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.proj = nn.Linear(SEQ_LEN*d_model, PRED_LEN)

    def forward(self, x_imf, x_raw, x_ex):
        imf_summary = x_imf.mean(dim=-1, keepdim=True)
        feats = torch.cat([x_raw, x_ex, imf_summary], dim=-1)
        x = self.enc_in(feats)
        attn, _ = self.attn(x, x, x)
        x = x + attn
        x = x + self.ff(x)
        x = x.reshape(x.size(0), -1)
        return self.proj(x).unsqueeze(-1)
