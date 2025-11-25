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
    def __init__(
        self,
        n_imfs,
        high_idx,
        mid_idx,
        seasonal_idx,
        trend_idx,
        scaler_raw=None,
        imf_feats=None,
        assign_temp=1.0,
    ):
        super().__init__()
        self.scaler_raw = scaler_raw
        self.n_imfs = n_imfs
        self.assign_temp = max(assign_temp, 1e-3)

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

        self.high_branch = TimesBlock(SEQ_LEN, PRED_LEN)
        self.mid_branch = SeasonalTCN(SEQ_LEN, PRED_LEN)
        self.season_branch = SeasonalTCN(SEQ_LEN, PRED_LEN)
        self.trend_branch = TrendLSTM(SEQ_LEN, PRED_LEN, in_channels=1)

        self.raw_m = RawEncoder()

        self.group_names = ["high", "mid", "season", "trend"]
        if self.imf_feat_dim > 0:
            self.high_assign = nn.Sequential(
                nn.Linear(self.imf_feat_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.mid_assign = nn.Sequential(
                nn.Linear(self.imf_feat_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.season_assign = nn.Sequential(
                nn.Linear(self.imf_feat_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.trend_assign = nn.Sequential(
                nn.Linear(self.imf_feat_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
        else:
            self.register_parameter("high_logits", nn.Parameter(torch.zeros(n_imfs, 1)))
            self.register_parameter("mid_logits", nn.Parameter(torch.zeros(n_imfs, 1)))
            self.register_parameter("season_logits", nn.Parameter(torch.zeros(n_imfs, 1)))
            self.register_parameter("trend_logits", nn.Parameter(torch.zeros(n_imfs, 1)))
            with torch.no_grad():
                if self.high_idx:
                    self.high_logits[self.high_idx] = 1.0
                if self.mid_idx:
                    self.mid_logits[self.mid_idx] = 1.0
                if self.season_idx:
                    self.season_logits[self.season_idx] = 1.0
                if self.trend_idx:
                    self.trend_logits[self.trend_idx] = 1.0

        self.align_high = nn.Linear(PRED_LEN, PRED_LEN)
        self.align_mid = nn.Linear(PRED_LEN, PRED_LEN)
        self.align_season = nn.Linear(PRED_LEN, PRED_LEN)
        self.align_trend = nn.Linear(PRED_LEN, PRED_LEN)
        self.group_gate = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.alpha_context = nn.GRU(1, 32, batch_first=True)
        self.alpha_gate = nn.Sequential(
            nn.Linear(32 + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def set_assign_temp(self, value):
        self.assign_temp = max(float(value), 1e-3)

    def _apply_branch(self, branch, series):
        B, L, N = series.shape
        flat = series.permute(0, 2, 1).reshape(B * N, L, 1)
        out = branch(flat)
        return out.view(B, N, PRED_LEN, 1)

    def _group_weights(self):
        weights = {}
        feats = self.imf_feats
        if self.imf_feat_dim > 0 and feats.numel() > 0:
            high_log = self.high_assign(feats).squeeze(-1)
            mid_log = self.mid_assign(feats).squeeze(-1)
            season_log = self.season_assign(feats).squeeze(-1)
            trend_log = self.trend_assign(feats).squeeze(-1)
        else:
            high_log = self.high_logits.squeeze(-1)
            mid_log = self.mid_logits.squeeze(-1)
            season_log = self.season_logits.squeeze(-1)
            trend_log = self.trend_logits.squeeze(-1)

        tau = self.assign_temp
        weights["high"] = torch.softmax(high_log / tau, dim=0)
        weights["mid"] = torch.softmax(mid_log / tau, dim=0)
        weights["season"] = torch.softmax(season_log / tau, dim=0)
        weights["trend"] = torch.softmax(trend_log / tau, dim=0)
        return weights

    def forward(self, x_imf, x_raw, x_ex):
        if x_imf.size(-1) == 0:
            raw_pred = self.raw_m(x_raw, x_ex)
            imf_pred = torch.zeros_like(raw_pred)
            return raw_pred, imf_pred, raw_pred

        weights = self._group_weights()

        high_all = self._apply_branch(self.high_branch, x_imf)
        mid_all = self._apply_branch(self.mid_branch, x_imf)
        season_all = self._apply_branch(self.season_branch, x_imf)
        trend_all = self._apply_branch(self.trend_branch, x_imf)

        high_pred = (high_all * weights["high"].view(1, -1, 1, 1)).sum(dim=1)
        mid_pred = (mid_all * weights["mid"].view(1, -1, 1, 1)).sum(dim=1)
        season_pred = (season_all * weights["season"].view(1, -1, 1, 1)).sum(dim=1)
        trend_pred = (trend_all * weights["trend"].view(1, -1, 1, 1)).sum(dim=1)

        def align_group(pred, align_layer):
            mean = pred.mean(dim=1, keepdim=True)
            std = pred.std(dim=1, keepdim=True) + 1e-6
            norm = (pred - mean) / std
            flat = norm.squeeze(-1)
            aligned = align_layer(flat).unsqueeze(-1)
            return aligned, mean.squeeze(-1), std.squeeze(-1)

        high_aligned, high_mean, high_std = align_group(high_pred, self.align_high)
        mid_aligned, mid_mean, mid_std = align_group(mid_pred, self.align_mid)
        season_aligned, season_mean, season_std = align_group(season_pred, self.align_season)
        trend_aligned, trend_mean, trend_std = align_group(trend_pred, self.align_trend)

        gate_input = torch.cat([
            high_mean, high_std,
            mid_mean, mid_std,
            season_mean, season_std,
            trend_mean, trend_std
        ], dim=-1)
        group_logits = self.group_gate(gate_input)
        B = gate_input.size(0)
        group_weights = torch.softmax(group_logits, dim=-1).view(B, 4, 1, 1)
        group_stack = torch.stack([high_aligned, mid_aligned, season_aligned, trend_aligned], dim=1)
        imf_pred = (group_weights * group_stack).sum(dim=1)

        raw_pred = self.raw_m(x_raw, x_ex)

        raw_mean = raw_pred.mean(dim=1, keepdim=True)
        raw_std = raw_pred.std(dim=1, keepdim=True) + 1e-6

        imf_mean = imf_pred.mean(dim=1, keepdim=True)
        imf_std = imf_pred.std(dim=1, keepdim=True) + 1e-6
        imf_centered = imf_pred - imf_mean
        imf_scaled = imf_centered / imf_std
        imf_flat = imf_scaled.squeeze(-1)
        imf_aligned = imf_flat.unsqueeze(-1)

        raw_stats = torch.cat([raw_mean.squeeze(-1), raw_std.squeeze(-1)], dim=-1)
        imf_stats = torch.cat([imf_mean.squeeze(-1), imf_std.squeeze(-1)], dim=-1)

        ctx, _ = self.alpha_context(x_raw)
        ctx_feat = ctx[:, -1]

        alpha_in = torch.cat([raw_stats, imf_stats, ctx_feat], dim=-1)
        alpha = self.alpha_gate(alpha_in).unsqueeze(-1) * 0.7

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
    def __init__(
        self,
        n_imfs,
        high_idx,
        mid_idx,
        seasonal_idx,
        trend_idx,
        scaler_raw=None,
        imf_feats=None,
        assign_temp=1.0,
    ):
        super().__init__()
        self.inner = IFMultiModel(
            n_imfs=n_imfs,
            high_idx=high_idx,
            mid_idx=mid_idx,
            seasonal_idx=seasonal_idx,
            trend_idx=trend_idx,
            scaler_raw=scaler_raw,
            imf_feats=imf_feats,
            assign_temp=assign_temp,
        )

    def forward(self, x_imf, x_raw, x_ex):
        final, _, _ = self.inner(x_imf, x_raw, x_ex)
        return final

    def set_assign_temp(self, value):
        self.inner.set_assign_temp(value)
