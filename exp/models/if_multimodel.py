import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from config import SEQ_LEN, PRED_LEN


def compute_imf_features(imfs):
    n_imfs = imfs.shape[1]
    feats = []

    for k in range(n_imfs):
        imf = imfs[:, k]
        fft_vals = np.abs(np.fft.rfft(imf))
        freqs = np.fft.rfftfreq(len(imf), d=1)

        dom = freqs[np.argmax(fft_vals)]
        centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9)
        entropy = -np.sum((fft_vals / np.sum(fft_vals)) * np.log((fft_vals / np.sum(fft_vals)) + 1e-9))
        rms = np.sqrt(np.mean(imf ** 2))

        feats.append([dom, centroid, entropy, rms])

    if feats:
        return np.array(feats, dtype=np.float32)
    return np.zeros((0, 4), dtype=np.float32)


def auto_group_imfs(imfs, n_groups=4):
    n_imfs = imfs.shape[1]
    if n_imfs == 0:
        return [], [], [], []
    n_groups = max(1, min(n_groups, n_imfs))

    feats = compute_imf_features(imfs)
    scaled_feats = StandardScaler().fit_transform(feats)

    kmeans = KMeans(n_clusters=n_groups, n_init=30)
    labels = kmeans.fit_predict(scaled_feats)
    groups = [[] for _ in range(n_groups)]

    for idx, g in enumerate(labels):
        groups[g].append(idx)

    empty_groups = [i for i, g in enumerate(groups) if len(g) == 0]
    while empty_groups:
        donor_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
        if len(groups[donor_idx]) <= 1:
            break
        target_idx = empty_groups.pop()
        groups[target_idx].append(groups[donor_idx].pop())

    groups = [g for g in groups if g]
    groups_sorted = sorted(groups, key=lambda g: np.mean([feats[i][0] for i in g]), reverse=True)

    while len(groups_sorted) < 4:
        groups_sorted.append([])

    return groups_sorted[0], groups_sorted[1], groups_sorted[2], groups_sorted[3]


class RawEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.GRU(5, hidden, batch_first=True)
        self.dec = nn.GRU(1, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x_raw, x_ex):
        x = torch.cat([x_raw, x_ex], dim=-1)
        _, h = self.enc(x)

        dec_in = torch.zeros(x.shape[0], PRED_LEN, 1, device=x.device)
        out, _ = self.dec(dec_in, h)
        return self.proj(out)


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64, heads=4):
        super().__init__()
        self.seq_len = seq_len

        self.emb = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        self.proj = nn.Linear(seq_len*d_model, pred_len)

    def forward(self, x):
        x = self.emb(x)

        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        x = x.reshape(x.size(0), -1)
        return self.proj(x).unsqueeze(-1)


class SeasonalTCN(nn.Module):
    def __init__(self, seq_len, pred_len, hidden=64):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        self.fc = nn.Linear(seq_len, pred_len)
        self.final = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return self.final(x)


class TrendLSTM(nn.Module):
    def __init__(self, seq_len, pred_len, hidden=64, in_channels=1):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, 2, batch_first=True, dropout=0.1)
        self.dec = nn.Linear(hidden, pred_len)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_last = self.ln(h[-1])
        return self.dec(h_last).unsqueeze(-1)


class IFMultiModel(nn.Module):
    def __init__(self, n_imfs, high_idx, mid_idx, seasonal_idx, trend_idx, scaler_raw=None, imf_feats=None):
        super().__init__()
        self.scaler_raw = scaler_raw

        self.high_idx = high_idx
        self.mid_idx = mid_idx
        self.season_idx = seasonal_idx
        self.trend_idx = trend_idx

        if imf_feats is None:
            imf_feats = np.zeros((n_imfs, 0), dtype=np.float32)
        if imf_feats.ndim == 1:
            imf_feats = imf_feats[np.newaxis, :]
        self.register_buffer("imf_feats", torch.tensor(imf_feats, dtype=torch.float32))
        self.imf_feat_dim = self.imf_feats.shape[1] if self.imf_feats.numel() > 0 else 0
        gate_extra_dim = self.imf_feat_dim

        if self.trend_idx:
            self.register_buffer("trend_idx_tensor", torch.tensor(self.trend_idx, dtype=torch.long))
        else:
            self.trend_idx_tensor = None

        self.high_m = nn.ModuleList([TimesBlock(SEQ_LEN, PRED_LEN) for _ in high_idx])
        self.mid_m = nn.ModuleList([SeasonalTCN(SEQ_LEN, PRED_LEN) for _ in mid_idx])
        self.season_m = nn.ModuleList([SeasonalTCN(SEQ_LEN, PRED_LEN) for _ in seasonal_idx])
        self.trend_m = TrendLSTM(SEQ_LEN, PRED_LEN, in_channels=len(trend_idx)) if trend_idx else None

        self.raw_m = RawEncoder()

        self.imf_gate = nn.Sequential(
            nn.Linear(PRED_LEN + gate_extra_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.alpha_gate = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.align = nn.Linear(PRED_LEN, PRED_LEN)

        self.reset_parameters()

    def forward(self, x_imf, x_raw, x_ex):
        preds = []
        feat_list = [] if self.imf_feat_dim > 0 else None
        for i, idx in enumerate(self.high_idx):
            preds.append(self.high_m[i](x_imf[:, :, idx:idx+1]))
            if feat_list is not None:
                feat_list.append(self.imf_feats[idx])
        for i, idx in enumerate(self.mid_idx):
            preds.append(self.mid_m[i](x_imf[:, :, idx:idx+1]))
            if feat_list is not None:
                feat_list.append(self.imf_feats[idx])
        for i, idx in enumerate(self.season_idx):
            preds.append(self.season_m[i](x_imf[:, :, idx:idx+1]))
            if feat_list is not None:
                feat_list.append(self.imf_feats[idx])

        if self.trend_m is not None and self.trend_idx_tensor is not None:
            trend_series = x_imf[:, :, self.trend_idx_tensor]
            preds.append(self.trend_m(trend_series))
            if feat_list is not None and self.trend_idx:
                trend_feat = self.imf_feats.index_select(0, self.trend_idx_tensor).mean(dim=0)
                feat_list.append(trend_feat)

        raw_pred = self.raw_m(x_raw, x_ex)

        if preds:
            preds_stack = torch.stack(preds, dim=1)
            gate_input = preds_stack.squeeze(-1)
            B, N, L = gate_input.shape

            if feat_list:
                feats_sel = torch.stack(feat_list, dim=0).to(gate_input.device)
                feats_sel = feats_sel.unsqueeze(0).expand(B, -1, -1)
                gate_feat = torch.cat([gate_input, feats_sel], dim=-1)
            else:
                gate_feat = gate_input

            gate_scores = self.imf_gate(gate_feat.view(B * N, -1)).view(B, N, 1, 1)
            weights = torch.softmax(gate_scores, dim=1)
            imf_pred = (weights * preds_stack).sum(dim=1)
        else:
            imf_pred = torch.zeros_like(raw_pred)
            final = raw_pred
            return final, imf_pred, raw_pred

        raw_mean = raw_pred.mean(dim=1, keepdim=True)
        raw_std = raw_pred.std(dim=1, keepdim=True) + 1e-6

        imf_mean = imf_pred.mean(dim=1, keepdim=True)
        imf_std = imf_pred.std(dim=1, keepdim=True) + 1e-6
        imf_centered = imf_pred - imf_mean
        imf_scaled = imf_centered / imf_std
        imf_flat = imf_scaled.squeeze(-1)
        imf_aligned = self.align(imf_flat).unsqueeze(-1)

        raw_stats = torch.cat([raw_mean.squeeze(-1), raw_std.squeeze(-1)], dim=-1)
        imf_stats = torch.cat([imf_mean.squeeze(-1), imf_std.squeeze(-1)], dim=-1)

        alpha_in = torch.cat([raw_stats, imf_stats], dim=-1)
        alpha = self.alpha_gate(alpha_in).unsqueeze(-1)

        final = alpha * raw_pred + (1 - alpha) * imf_aligned

        return final, imf_pred, raw_pred

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)


class IFModel(BaseModel):
    def __init__(self, n_imfs, high_idx, mid_idx, seasonal_idx, trend_idx, scaler_raw=None, imf_feats=None):
        super().__init__()
        self.inner = IFMultiModel(
            n_imfs=n_imfs,
            high_idx=high_idx,
            mid_idx=mid_idx,
            seasonal_idx=seasonal_idx,
            trend_idx=trend_idx,
            scaler_raw=scaler_raw,
            imf_feats=imf_feats
        )

    def forward(self, x_imf, x_raw, x_ex):
        final, _, _ = self.inner(x_imf, x_raw, x_ex)
        return final
