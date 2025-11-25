import torch
import torch.nn as nn
from config import LR

class UnifiedTrainer:
    def __init__(self, model, train_loader, val_loader, device, scaler_raw=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.crit = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', patience=5, factor=0.5
        )

        if scaler_raw is not None and hasattr(scaler_raw, "scale_") and hasattr(scaler_raw, "mean_"):
            scale = torch.tensor(scaler_raw.scale_, dtype=torch.float32, device=self.device)
            mean = torch.tensor(scaler_raw.mean_, dtype=torch.float32, device=self.device)
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
        for x_imf, x_raw, x_ex, y in self.train_loader:
            x_imf, x_raw, x_ex, y = (
                x_imf.to(self.device),
                x_raw.to(self.device),
                x_ex.to(self.device),
                y.to(self.device)
            )

            pred = self.model(x_imf, x_raw, x_ex)
            loss = self.crit(pred, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total = 0
        with torch.no_grad():
            for x_imf, x_raw, x_ex, y in self.val_loader:
                x_imf, x_raw, x_ex, y = (
                    x_imf.to(self.device),
                    x_raw.to(self.device),
                    x_ex.to(self.device),
                    y.to(self.device)
                )
                pred = self.model(x_imf, x_raw, x_ex)
                pred_denorm = self._inverse_transform(pred)
                target_denorm = self._inverse_transform(y)
                loss = self.crit(pred_denorm, target_denorm)
                total += loss.item()
        return total / len(self.val_loader)
