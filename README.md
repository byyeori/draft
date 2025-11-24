# IF-IMF Time Series Forecasting Framework

## Project Structure

```
project/
├── decomposition/
│   ├── __init__.py
│   ├── if_model.py              # IF decomposition
│   └── imf_analyzer.py          # IMF feature extraction & grouping
│
├── models/
│   ├── __init__.py
│   ├── group_models.py          # Model selector
│   ├── cnn_hf.py                # CNN for HF
│   ├── timesblock.py            # TimesBlock for Mid
│   ├── timesnet.py              # TimesNet for Seasonal
│   ├── lstm_trend.py            # LSTM for Trend
│   ├── linear_trend.py          # Linear for Trend
│   └── fusion.py                # Weighted/Gated fusion
│
├── layers/
│   ├── __init__.py
│   ├── freq_loss.py             # Frequency domain loss
│   └── fft_layer.py             # FFT operations
│
├── utils/
│   ├── __init__.py
│   ├── grouping_config.yaml     # IMF grouping rules
│   ├── fusion_config.yaml       # Fusion & lambda config
│   └── data_loader.py           # Dataset loader
│
├── scripts/
│   ├── run_if.py                # Step 1: IF decomposition
│   ├── run_grouping.py          # Step 2: IMF grouping
│   └── run_train.py             # Step 3: Training
│
├── experiments/
│   ├── exp_main.py              # Main pipeline
│   ├── exp_if_vs_vmd.py         # IF vs VMD comparison
│   ├── exp_grouping_test.py     # Rule vs Clustering
│   ├── exp_model_ablation.py    # Model ablation
│   ├── exp_fusion_test.py       # Fusion comparison
│   └── exp_freq_loss.py         # Frequency loss ablation
│
├── data/
│   ├── raw/                     # Raw time series
│   ├── imfs/                    # Saved IMF components
│   └── processed/               # Processed datasets
│
├── checkpoints/                 # Model checkpoints
├── results/                     # Experiment results
├── run.sh                       # Full pipeline script
└── requirements.txt
```

## Pipeline Flow

```
Raw Time Series
      ↓
[1] IF Decomposition (offline)
      ↓
IMF₁ ~ IMFₖ (.npy)
      ↓
[2] IMF Feature Extraction
    - FFT dominant frequency
    - Spectral entropy
    - Zero-crossing rate
    - Periodicity (autocorr)
    - Trend strength (R²)
      ↓
[3] Automatic Grouping
    - Rule-based OR
    - K-Means clustering
      ↓
Groups: {HF, Mid, Seasonal, Trend}
      ↓
[4] Group-Specific Models
    - HF → CNN
    - Mid → TimesBlock
    - Seasonal → TimesNet
    - Trend → LSTM/Linear
      ↓
[5] Fusion (Weighted/Gated)
      ↓
[6] Loss = Time Loss + λ * Frequency Loss
      ↓
Final Prediction
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
bash run.sh
```

### 3. Step-by-Step Execution
```bash
# Step 1: IF Decomposition
python scripts/run_if.py --data data/raw/your_data.csv

# Step 2: IMF Grouping
python scripts/run_grouping.py --method rule  # or kmeans

# Step 3: Training
python scripts/run_train.py --config utils/fusion_config.yaml
```

## Key Features

### 1. IF Decomposition
- Stable decomposition without hyperparameter sensitivity
- Offline preprocessing for efficiency
- Saves IMFs as .npy files

### 2. Automatic IMF Grouping
- **Rule-based**: Using frequency/periodicity thresholds
- **K-Means**: Clustering based on IMF features
- **Data Leakage Prevention**: Fit on train only

### 3. Group-Specific Models
| Group    | Model        | Purpose              |
|----------|--------------|----------------------|
| HF       | CNN          | High-freq noise      |
| Mid      | TimesBlock   | Mid-range patterns   |
| Seasonal | TimesNet     | Periodic components  |
| Trend    | LSTM/Linear  | Long-term trends     |

### 4. Frequency Loss (FreDF Style)
```python
L_total = L_time + λ * L_freq

L_freq = L_amplitude + 0.5 * L_phase
```

### 5. Adaptive Lambda (λ)
| Group    | λ Value   | Reason                    |
|----------|-----------|---------------------------|
| HF       | 0.0-0.1   | Focus on time domain      |
| Mid      | 0.1-0.3   | Balanced                  |
| Seasonal | 0.5-1.0   | Strong freq structure     |
| Trend    | 0.2-0.5   | Moderate freq importance  |

## Configuration

### grouping_config.yaml
```yaml
rule_based:
  hf:
    zcr_min: 0.3
    entropy_min: 0.7
    periodicity_max: 0.2
  seasonal:
    periodicity_min: 0.6
  trend:
    trend_strength_min: 0.7
    zcr_max: 0.1

kmeans:
  n_clusters: 4
  random_state: 42
  features: [dominant_freq, zcr, periodicity, trend_strength]
```

### fusion_config.yaml
```yaml
fusion_type: weighted  # or gated

lambda_weights:
  HF: 0.05
  Mid: 0.2
  Seasonal: 0.8
  Trend: 0.3

time_loss: mse  # or mae
freq_loss_weight: 1.0
```

## Experiments

All experiments are in `experiments/` directory:

1. **exp_main.py**: Full pipeline validation
2. **exp_if_vs_vmd.py**: IF vs VMD decomposition
3. **exp_grouping_test.py**: Rule-based vs K-Means
4. **exp_model_ablation.py**: Test different model combinations
5. **exp_fusion_test.py**: Weighted vs Gated fusion
6. **exp_freq_loss.py**: Time-only vs Time+Freq loss

## Important Notes

⚠️ **Data Leakage Prevention**
- K-Means clustering: Fit on train IMFs only
- Use `.predict()` for validation/test IMFs
- Never use test data statistics in grouping

⚠️ **Trend Overfitting**
- Keep trend models simple (Linear/shallow LSTM)
- Use strong regularization
- Monitor validation loss carefully

⚠️ **Frequency Loss**
- Include both amplitude and phase information
- Use `torch.fft.rfft` for real-valued signals
- Normalize frequency components if needed

## Citation

If you use this framework, please cite:
```
[Your paper citation here]
```

## License

MIT License