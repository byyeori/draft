"""
Group-Specific Model Selector
Maps IMF groups to appropriate forecasting models
"""

import torch
import torch.nn as nn


# ==================== HF Group: CNN Model ====================
class CNNHF(nn.Module):
    """
    CNN for High-Frequency components
    Captures local patterns and rapid oscillations
    """
    
    def __init__(self, seq_len, pred_len, d_model=64, kernel_sizes=[3, 5, 7]):
        super(CNNHF, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Multi-scale CNN
        self.conv_blocks = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(1, d_model, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.BatchNorm1d(d_model)
            )
            self.conv_blocks.append(conv)
        
        # Fusion and projection
        self.fusion = nn.Conv1d(d_model * len(kernel_sizes), d_model, 1)
        self.projection = nn.Linear(seq_len, pred_len)
        self.output = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x.transpose(1, 2)  # [batch, 1, seq_len]
        
        # Multi-scale features
        features = []
        for conv_block in self.conv_blocks:
            feat = conv_block(x)  # [batch, d_model, seq_len]
            features.append(feat)
        
        # Concatenate multi-scale features
        x = torch.cat(features, dim=1)  # [batch, d_model*3, seq_len]
        
        # Fusion
        x = self.fusion(x)  # [batch, d_model, seq_len]
        
        # Temporal projection
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        x = self.projection(x.transpose(1, 2))  # [batch, d_model, pred_len]
        x = x.transpose(1, 2)  # [batch, pred_len, d_model]
        
        # Output
        x = self.output(x)  # [batch, pred_len, 1]
        
        return x


# ==================== Mid Group: TimesBlock ====================
class TimesBlock(nn.Module):
    """
    TimesBlock for Mid-frequency components
    Captures medium-range temporal patterns
    """
    
    def __init__(self, seq_len, pred_len, d_model=64, d_ff=128, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Inception-like block
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, d_model, kernel_size=5, padding=2)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        
        # Temporal attention
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Projection
        self.projection = nn.Linear(seq_len, pred_len)
        self.output = nn.Linear(d_model, 1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x_input = x.transpose(1, 2)  # [batch, 1, seq_len]
        
        # Inception block
        c1 = self.conv1(x_input)
        c3 = self.conv3(x_input)
        c5 = self.conv5(x_input)
        
        x = torch.cat([c1, c3, c5], dim=1)  # [batch, d_model*3, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model*3]
        
        # Feed-forward
        x = self.ff(x)  # [batch, seq_len, d_model]
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Projection
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.projection(x)  # [batch, d_model, pred_len]
        x = x.transpose(1, 2)  # [batch, pred_len, d_model]
        
        # Output
        x = self.output(x)  # [batch, pred_len, 1]
        
        return x


# ==================== Seasonal Group: TimesNet-style ====================
class SeasonalNet(nn.Module):
    """
    Seasonal pattern extractor
    Focuses on periodic components
    """
    
    def __init__(self, seq_len, pred_len, d_model=64, period=24):
        super(SeasonalNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        
        # Seasonal decomposition
        self.seasonal_conv = nn.Conv1d(1, d_model, kernel_size=period, 
                                      stride=1, padding=period//2)
        
        # TCN-like structure
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model)
        )
        
        # Projection
        self.projection = nn.Linear(seq_len, pred_len)
        self.output = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x.transpose(1, 2)  # [batch, 1, seq_len]
        
        # Extract seasonal patterns
        x = self.seasonal_conv(x)  # [batch, d_model, seq_len]
        
        # TCN processing
        x = self.tcn(x)  # [batch, d_model, seq_len]
        
        # Projection
        x = self.projection(x)  # [batch, d_model, pred_len]
        x = x.transpose(1, 2)  # [batch, pred_len, d_model]
        
        # Output
        x = self.output(x)  # [batch, pred_len, 1]
        
        return x


# ==================== Trend Group: LSTM ====================
class TrendLSTM(nn.Module):
    """
    LSTM for Trend components
    Captures long-term dependencies (kept simple to avoid overfitting)
    """
    
    def __init__(self, seq_len, pred_len, d_model=32, num_layers=2):
        super(TrendLSTM, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.projection = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, d_model]
        
        # Use last output
        last_out = lstm_out[:, -1, :]  # [batch, d_model]
        
        # Project to prediction length
        out = self.projection(last_out)  # [batch, pred_len]
        out = out.unsqueeze(-1)  # [batch, pred_len, 1]
        
        return out


# ==================== Trend Group: Linear (Alternative) ====================
class TrendLinear(nn.Module):
    """
    Simple Linear model for Trend
    Very stable, prevents overfitting
    """
    
    def __init__(self, seq_len, pred_len):
        super(TrendLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x.squeeze(-1)  # [batch, seq_len]
        out = self.linear(x)  # [batch, pred_len]
        out = out.unsqueeze(-1)  # [batch, pred_len, 1]
        
        return out


# ==================== Model Factory ====================
def get_model_for_group(group_name, seq_len, pred_len, model_config=None):
    """
    Get appropriate model for IMF group
    
    Args:
        group_name: 'HF', 'Mid', 'Seasonal', or 'Trend'
        seq_len: Input sequence length
        pred_len: Prediction length
        model_config: Optional configuration dict
        
    Returns:
        PyTorch model instance
    """
    if model_config is None:
        model_config = {}
    
    if group_name == 'HF':
        model = CNNHF(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=model_config.get('d_model', 64),
            kernel_sizes=model_config.get('kernel_sizes', [3, 5, 7])
        )
    
    elif group_name == 'Mid':
        model = TimesBlock(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=model_config.get('d_model', 64),
            d_ff=model_config.get('d_ff', 128)
        )
    
    elif group_name == 'Seasonal':
        model = SeasonalNet(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=model_config.get('d_model', 64),
            period=model_config.get('period', 24)
        )
    
    elif group_name == 'Trend':
        trend_type = model_config.get('trend_type', 'linear')
        
        if trend_type == 'lstm':
            model = TrendLSTM(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=model_config.get('d_model', 32),
                num_layers=model_config.get('num_layers', 2)
            )
        else:  # linear
            model = TrendLinear(seq_len=seq_len, pred_len=pred_len)
    
    else:
        raise ValueError(f"Unknown group: {group_name}")
    
    return model


if __name__ == "__main__":
    # Test all models
    batch_size = 16
    seq_len = 96
    pred_len = 24
    
    print("Testing Group-Specific Models")
    print("=" * 60)
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, 1)
    
    for group in ['HF', 'Mid', 'Seasonal', 'Trend']:
        print(f"\nTesting {group} model:")
        model = get_model_for_group(group, seq_len, pred_len)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters:   {n_params:,}")
    
    print("\n" + "=" * 60)