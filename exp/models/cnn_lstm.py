import torch
import torch.nn as nn
from .base_model import BaseModel
from config import SEQ_LEN, PRED_LEN


class CNNLSTMModel(BaseModel):
    """
    CNN-LSTM baseline under controlled experiment.
    - CNN extracts short-term local features from raw sequence.
    - LSTM captures longer temporal dependency.
    - Final linear layer generates prediction horizon.
    """
    def __init__(self, cnn_channels=32, lstm_hidden=64):
        super().__init__()

        # 1D CNN block
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Final projection
        self.fc = nn.Linear(lstm_hidden, PRED_LEN)

    def forward(self, x_imf, x_raw, x_ex):
        """
        x_raw: (B, L, 1)
        """
        # CNN requires (B, C, L)
        x = x_raw.transpose(1, 2)    # (B, 1, L)
        x = self.conv(x)             # (B, C, L)

        # LSTM requires (B, L, C)
        x = x.transpose(1, 2)        # (B, L, C)

        _, (h, _) = self.lstm(x)
        h_last = h[-1]               # (B, hidden)

        out = self.fc(h_last)        # (B, pred_len)
        return out.unsqueeze(-1)     # (B, pred_len, 1)
