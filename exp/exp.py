import argparse
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import *
from utils.seed import set_seed
from utils.loader import load_data
from dataset import UnifiedDataset
from trainer import UnifiedTrainer

# Models
from models.if_multimodel import (
    IFModel,
    auto_group_imfs,
    compute_imf_features
)
from models.timesnet import TimesNetModel
from models.informer import InformerModel
from models.tft import TFTModel


def compute_assign_temp(epoch, start, end, warmup, decay):
    start = float(start)
    end = float(end)
    if decay <= 0 or start == end:
        return end
    if epoch < warmup:
        return start
    progress = min(max(epoch - warmup, 0) / decay, 1.0)
    return start + (end - start) * progress


def evaluate_on_test(model, loader, device, scaler_raw):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x_imf, x_raw, x_ex, y in loader:
            x_imf, x_raw, x_ex, y = (
                x_imf.to(device),
                x_raw.to(device),
                x_ex.to(device),
                y.to(device)
            )
            out = model(x_imf, x_raw, x_ex)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, 1)
    trues = np.concatenate(trues, axis=0).reshape(-1, 1)

    preds_denorm = scaler_raw.inverse_transform(preds).reshape(-1)
    trues_denorm = scaler_raw.inverse_transform(trues).reshape(-1)

    diff = preds_denorm - trues_denorm
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def run_experiment(name, model_factory, train_loader, val_loader, test_loader, device, scaler_raw, assign_sched=None):
    print(f"\n===== Running {name} =====")
    model = model_factory().to(device)
    trainer = UnifiedTrainer(model, train_loader, val_loader, device, scaler_raw)

    best = float('inf')
    ckpt_path = f"best_{name}.pth"

    for ep in range(EPOCHS):
        if assign_sched and hasattr(model, "set_assign_temp"):
            tau = compute_assign_temp(ep, assign_sched["start"], assign_sched["end"], assign_sched["warmup"], assign_sched["decay"])
            model.set_assign_temp(tau)
        tr = trainer.train_epoch()
        v = trainer.validate()
        print(f"[{name}] Epoch {ep+1}: Train={tr:.5f} | Val={v:.5f}")

        trainer.scheduler.step(v)

        if v < best:
            best = v
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if assign_sched and hasattr(model, "set_assign_temp"):
        model.set_assign_temp(assign_sched["end"])
    metrics = evaluate_on_test(model, test_loader, device, scaler_raw)
    print(f"→ Best {name} Val = {best:.5f} | Test MSE={metrics['MSE']:.5f} MAE={metrics['MAE']:.5f} RMSE={metrics['RMSE']:.5f}")
    return best, metrics


def main(argv=None):
    base_dir = Path(__file__).resolve().parent
    default_data = base_dir.parent / "data"

    parser = argparse.ArgumentParser(description="Unified IMF experiments")
    parser.add_argument("--user", type=str, default="kdy", help="User name used to choose default CSVs")
    parser.add_argument("--data-dir", type=str, default=str(default_data), help="Directory containing IFs_*.csv and timedata_*.csv")
    parser.add_argument("--imf-path", type=str, default=None, help="Override IMF CSV path")
    parser.add_argument("--raw-path", type=str, default=None, help="Override raw CSV path")
    parser.add_argument("--if-assign-temp", type=float, default=None, help="(Deprecated) Use a fixed assignment temperature for IF model")
    parser.add_argument("--if-assign-temp-start", type=float, default=2.0, help="Starting soft assignment temperature for IF model")
    parser.add_argument("--if-assign-temp-end", type=float, default=0.2, help="Ending soft assignment temperature for IF model")
    parser.add_argument("--if-assign-warmup", type=int, default=10, help="Number of warmup epochs before annealing IF assignment temperature")
    parser.add_argument("--if-assign-decay", type=int, default=40, help="Epochs over which to anneal IF assignment temperature")
    args = parser.parse_args(argv)

    set_seed(SEED)

    if args.if_assign_temp is not None:
        assign_start = args.if_assign_temp
        assign_end = args.if_assign_temp
        assign_warmup = 0
        assign_decay = 0
    else:
        assign_start = args.if_assign_temp_start
        assign_end = args.if_assign_temp_end
        assign_warmup = args.if_assign_warmup
        assign_decay = args.if_assign_decay

    assign_sched = {
        "start": assign_start,
        "end": assign_end,
        "warmup": assign_warmup,
        "decay": assign_decay
    }

    data_dir = Path(args.data_dir)
    imf_path = Path(args.imf_path) if args.imf_path else data_dir / f"IFs_{args.user.lower()}.csv"
    raw_path = Path(args.raw_path) if args.raw_path else data_dir / f"timedata_{args.user.lower()}.csv"

    if not imf_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing data files: {imf_path}, {raw_path}")

    imfs, raw = load_data(imf_path, raw_path)

    # Train/Val/Test split
    T = len(raw)
    tr_end = int(T*0.7)
    val_end = int(T*0.85)

    train_imf, train_raw = imfs[:tr_end], raw[:tr_end]
    val_imf,   val_raw   = imfs[tr_end:val_end], raw[tr_end:val_end]
    test_imf,  test_raw  = imfs[val_end:], raw[val_end:]

    # Scaling
    scaler_imf = StandardScaler()
    scaler_raw = StandardScaler()
    scaler_imf.fit(train_imf)
    scaler_raw.fit(train_raw.reshape(-1,1))

    # Dataset
    train_ds = UnifiedDataset(train_imf, train_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)
    val_ds   = UnifiedDataset(val_imf, val_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)
    test_ds  = UnifiedDataset(test_imf, test_raw, SEQ_LEN, PRED_LEN, scaler_imf, scaler_raw)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH)
    test_loader  = DataLoader(test_ds, batch_size=BATCH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    high_idx, mid_idx, season_idx, trend_idx = auto_group_imfs(train_imf)
    feats_raw = compute_imf_features(train_imf)
    if feats_raw.shape[0] > 0:
        feat_scaler = StandardScaler()
        feats_gate = feat_scaler.fit_transform(feats_raw)
    else:
        feats_gate = feats_raw

    def build_if():
        return IFModel(
            n_imfs=train_imf.shape[1],
            high_idx=high_idx,
            mid_idx=mid_idx,
            seasonal_idx=season_idx,
            trend_idx=trend_idx,
            scaler_raw=scaler_raw,
            imf_feats=feats_gate,
            assign_temp=assign_start
        )

    experiments = [
        ("IF", build_if, assign_sched),
        ("TimesNet", lambda: TimesNetModel(), None),
        ("Informer", lambda: InformerModel(), None),
        ("TFT", lambda: TFTModel(), None),
    ]

    results = {}
    for name, factory, sched in experiments:
        val_loss, metrics = run_experiment(name, factory, train_loader, val_loader, test_loader, device, scaler_raw, sched if sched is not None else None)
        results[name] = {"val": val_loss, **metrics}

    print("\n===== Final Comparison =====")
    rows = []
    for name, stats in results.items():
        print(f"{name}: Val={stats['val']:.5f} | MSE={stats['MSE']:.5f} | MAE={stats['MAE']:.5f} | RMSE={stats['RMSE']:.5f}")
        rows.append({
            "model": name,
            "val_loss": stats["val"],
            "mse": stats["MSE"],
            "mae": stats["MAE"],
            "rmse": stats["RMSE"]
        })

    metrics_df = pd.DataFrame(rows)
    user_tag = (args.user or "default").lower().replace(" ", "_")
    metrics_name = f"experiment_metrics_{user_tag}.csv"
    metrics_df.to_csv(metrics_name, index=False)
    print(f"\nSaved metrics to {metrics_name}")


if __name__ == "__main__":
    main()
