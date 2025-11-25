import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List
import os
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


USER_NAME = input("user: ")
IMF_PATH = os.path.join(f"IFs_{USER_NAME.lower()}.csv")
DATA_PATH = os.path.join(f"timedata_{USER_NAME.lower()}.csv")
SEQ_LEN = 96
PRED_LEN = 3
BATCH = 128
EPOCHS = 100
SEED = 1337
ASSIGN_TEMP_START = 2.0
ASSIGN_TEMP_END = 0.2
ASSIGN_TEMP_WARMUP = 10
ASSIGN_TEMP_DECAY = 40

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# def auto_group_imfs(imfs):
#     ... legacy reference ...


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


def compute_assign_temp(epoch):
    if ASSIGN_TEMP_START == ASSIGN_TEMP_END:
        return ASSIGN_TEMP_START
    if ASSIGN_TEMP_DECAY <= 0:
        return ASSIGN_TEMP_END
    if epoch < ASSIGN_TEMP_WARMUP:
        return ASSIGN_TEMP_START
    progress = min(max(epoch - ASSIGN_TEMP_WARMUP, 0) / ASSIGN_TEMP_DECAY, 1.0)
    return ASSIGN_TEMP_START + (ASSIGN_TEMP_END - ASSIGN_TEMP_START) * progress


class IFDataset(Dataset):
    def __init__(self, imfs, raw, seq_len, pred_len, scaler_imf, scaler_raw):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.imfs = scaler_imf.transform(imfs)
        self.raw = scaler_raw.transform(raw.reshape(-1, 1)).reshape(-1)
        self.n_imfs = self.imfs.shape[1]

        T = len(raw)
        hours = np.arange(T) % 24
        days = np.arange(T) % 7

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
        x_ex = self.exog[idx:idx+self.seq_len]
        y_raw = self.raw[idx+self.seq_len:idx+self.seq_len+self.pred_len]

        return (
            torch.FloatTensor(x_imf),
            torch.FloatTensor(x_raw).unsqueeze(-1),
            torch.FloatTensor(x_ex),
            torch.FloatTensor(y_raw).unsqueeze(-1)
        )


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
    def __init__(self, n_imfs, high_idx, mid_idx, seasonal_idx, trend_idx, scaler_raw, imf_feats):
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

        # self.align = nn.Linear(1, 1)
        self.align = nn.Linear(PRED_LEN, PRED_LEN)

        self.reset_parameters()

    def set_assign_temp(self, value):
        self.assign_temp = max(float(value), 1e-3)


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
            preds_stack = torch.stack(preds, dim=1)  # (B, n_imfs, L, 1)
            gate_input = preds_stack.squeeze(-1)     # (B, n_imfs, L)
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



class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.crit = nn.MSELoss()
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', patience=5, factor=0.5
        )

        scaler = getattr(model, "scaler_raw", None)
        if scaler is not None and hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
            scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)
            mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
            self._scale = scale.view(1, 1, 1)
            self._mean = mean.view(1, 1, 1)
        else:
            self._scale = None
            self._mean = None

    def _inverse_transform(self, tensor):
        if self._scale is None or self._mean is None:
            return tensor
        return tensor * self._scale + self._mean

    def train_epoch(self):
        self.model.train()
        total = 0

        for x_imf, x_raw, x_ex, y_raw in self.train_loader:
            x_imf = x_imf.to(self.device)
            x_raw = x_raw.to(self.device)
            x_ex  = x_ex.to(self.device)
            y_raw = y_raw.to(self.device)

            final, _, _ = self.model(x_imf, x_raw, x_ex)
            # final_np = final.detach().cpu().numpy()      # (B, L, 1)
            # final_np = final_np.reshape(-1, 1)
            # final_raw = self.model.scaler_raw.inverse_transform(final_np)
            # y_np = y_raw.cpu().numpy().reshape(-1,1)
            # y_raw_true = self.model.scaler_raw.inverse_transform(y_np)
            # loss_value = np.mean((final_raw - y_raw_true)**2)
            # loss = torch.tensor(loss_value, dtype=torch.float32, device=self.device)
            loss = self.crit(final, y_raw)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()

        return total / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total = 0

        with torch.no_grad():
            for x_imf, x_raw, x_ex, y_raw in self.val_loader:
                x_imf = x_imf.to(self.device)
                x_raw = x_raw.to(self.device)
                x_ex  = x_ex.to(self.device)
                y_raw = y_raw.to(self.device)

                final, _, _ = self.model(x_imf, x_raw, x_ex)

                final_denorm = self._inverse_transform(final)
                target_denorm = self._inverse_transform(y_raw)
                loss = self.crit(final_denorm, target_denorm)
                total += loss.item()

        return total / len(self.val_loader)



    def test(self, test_loader, save_path="test_predictions.csv"):
        self.model.eval()
        preds_all = []
        trues_all = []

        sample_count = 0
        mse_total = 0
        mae_total = 0

        with torch.no_grad():
            for x_imf, x_raw, x_ex, y_raw in test_loader:
                x_imf = x_imf.to(self.device)
                x_raw = x_raw.to(self.device)
                x_ex  = x_ex.to(self.device)
                y_raw = y_raw.to(self.device)

                final, _, _ = self.model(x_imf, x_raw, x_ex)
                final_np = final.cpu().numpy().squeeze(-1)  # (B, L)
                B, L = final_np.shape
                final_raw = self.model.scaler_raw.inverse_transform(final_np.reshape(-1, 1)).reshape(B, L)

                y_np = y_raw.cpu().numpy().squeeze(-1)
                y_raw_true = self.model.scaler_raw.inverse_transform(y_np.reshape(-1, 1)).reshape(B, L)

                diff = final_raw - y_raw_true
                mse_total += (diff**2).sum()
                mae_total += np.abs(diff).sum()
                sample_count += diff.size

                preds_all.append(final_raw)
                trues_all.append(y_raw_true)

        mse = mse_total / sample_count
        mae = mae_total / sample_count
        rmse = np.sqrt(mse)

        preds_all = np.concatenate(preds_all, axis=0).reshape(-1)
        trues_all = np.concatenate(trues_all, axis=0).reshape(-1)

        pd.DataFrame({"y_true": trues_all, "y_pred": preds_all}).to_csv(save_path, index=False)

        print(f"\n[Saved] Test predictions → {save_path}")

        return mse, mae, rmse


if __name__ == "__main__":
    full_imf = pd.read_csv(IMF_PATH, header=0).values.astype(np.float32)
    full_raw = pd.read_csv(DATA_PATH).values[:,0].astype(np.float32)

    total_len = len(full_raw)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    train_raw = full_raw[:train_end]
    val_raw   = full_raw[train_end:val_end]
    test_raw  = full_raw[val_end:]

    train_imf = full_imf[:train_end]
    val_imf   = full_imf[train_end:val_end]
    test_imf  = full_imf[val_end:]

    high_idx, mid_idx, season_idx, trend_idx = auto_group_imfs(train_imf)

    scaler_imf = StandardScaler()
    scaler_raw = StandardScaler()
    scaler_imf.fit(train_imf)
    scaler_raw.fit(train_raw.reshape(-1,1))

    train_dataset = IFDataset(train_imf, train_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)
    val_dataset   = IFDataset(val_imf, val_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)
    test_dataset  = IFDataset(test_imf, test_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    feats_raw = compute_imf_features(train_imf)
    if feats_raw.shape[0] > 0:
        feat_scaler = StandardScaler()
        feats_gate = feat_scaler.fit_transform(feats_raw)
    else:
        feats_gate = feats_raw

    print("IMF 그룹핑 결과:")
    print("High:", high_idx)
    print("Mid:", mid_idx)
    print("Seasonal:", season_idx)
    print("Trend:", trend_idx)

    doms = feats_raw[:, 0] if feats_raw.shape[0] > 0 else np.array([])
    for name, group in zip(
        ["High", "Mid", "Seasonal", "Trend"],
        [high_idx, mid_idx, season_idx, trend_idx]
    ):
        if group and doms.size:
            print(f"{name} dom mean: {doms[group].mean():.4f}")
        else:
            print(f"{name}: empty")

    model = IFMultiModel(
        train_dataset.n_imfs,
        high_idx,
        mid_idx,
        season_idx,
        trend_idx,
        scaler_raw,
        feats_gate,
        assign_temp=ASSIGN_TEMP_START
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, train_loader, val_loader, device)

    best = float('inf')
    patience = 20
    count = 0
    for epoch in range(EPOCHS):
        if hasattr(model, "set_assign_temp"):
            tau = compute_assign_temp(epoch)
            model.set_assign_temp(tau)

        tr = trainer.train_epoch()
        val = trainer.validate()
        print(f"Epoch {epoch+1}/{EPOCHS} - Train MSE: {tr:.6f} | Val MSE: {val:.6f}")
        trainer.scheduler.step(val)

        if val < best:
            best = val
            count = 0
            torch.save(model.state_dict(), "best.pth")
        else:
            count += 1

        if count >= patience:
            print("Early stopping")
            break

    print("\n학습 완료!")

    model.load_state_dict(torch.load("best.pth", map_location=device))
    if hasattr(model, "set_assign_temp"):
        model.set_assign_temp(ASSIGN_TEMP_END)

    test_mse, test_mae, test_rmse = trainer.test(test_loader)
    print(f"[TEST] MSE={test_mse:.5f} | MAE={test_mae:.5f} | RMSE={test_rmse:.5f}")
