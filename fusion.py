"""
Fusion Module and Frequency-Domain Loss
Combines group predictions and computes time+frequency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Fusion Strategies ====================

class WeightedFusion(nn.Module):
    """
    Weighted Fusion: y = Σ(w[g] * y_g)
    Learnable scalar weights for each group
    """
    
    def __init__(self, num_groups=4):
        super(WeightedFusion, self).__init__()
        
        # Learnable weights (initialized uniformly)
        self.weights = nn.Parameter(torch.ones(num_groups) / num_groups)
        
    def forward(self, group_predictions):
        """
        Args:
            group_predictions: List of [batch, pred_len, 1] tensors
            
        Returns:
            Fused prediction: [batch, pred_len, 1]
        """
        # Normalize weights with softmax
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum
        fused = torch.zeros_like(group_predictions[0])
        for i, pred in enumerate(group_predictions):
            if i < len(normalized_weights):
                fused += normalized_weights[i] * pred
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated Fusion: gate = sigmoid(Wx + b), y = Σ(gate[g] * y_g)
    Context-dependent fusion weights
    """
    
    def __init__(self, num_groups=4, d_model=64):
        super(GatedFusion, self).__init__()
        self.num_groups = num_groups
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_groups),
            nn.Sigmoid()
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, group_predictions):
        """
        Args:
            group_predictions: List of [batch, pred_len, 1] tensors
            
        Returns:
            Fused prediction: [batch, pred_len, 1]
        """
        # Stack predictions
        stacked = torch.stack(group_predictions[:self.num_groups], dim=1)
        # [batch, num_groups, pred_len, 1]
        
        # Compute context from average prediction
        avg_pred = torch.mean(stacked, dim=1)  # [batch, pred_len, 1]
        context = self.context_encoder(avg_pred.transpose(1, 2))  # [batch, d_model, 1]
        context = context.squeeze(-1)  # [batch, d_model]
        
        # Compute gates
        gates = self.gate_network(context)  # [batch, num_groups]
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # [batch, num_groups, 1, 1]
        
        # Apply gates
        fused = (stacked * gates).sum(dim=1)  # [batch, pred_len, 1]
        
        return fused


# ==================== Frequency Domain Loss ====================

class FrequencyLoss(nn.Module):
    """
    Frequency-Domain Loss (FreDF style)
    Compares amplitude and phase in frequency domain
    """
    
    def __init__(self, amp_weight=1.0, phase_weight=0.5):
        super(FrequencyLoss, self).__init__()
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        
    def forward(self, pred, true):
        """
        Args:
            pred: Predicted signal [batch, seq_len, 1] or [batch, seq_len]
            true: Ground truth signal [batch, seq_len, 1] or [batch, seq_len]
            
        Returns:
            Frequency domain loss (scalar)
        """
        # Ensure 2D: [batch, seq_len]
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if true.dim() == 3:
            true = true.squeeze(-1)
        
        # Apply FFT
        pred_fft = torch.fft.rfft(pred, dim=-1)  # Complex tensor
        true_fft = torch.fft.rfft(true, dim=-1)
        
        # Amplitude loss
        pred_amp = torch.abs(pred_fft)
        true_amp = torch.abs(true_fft)
        amp_loss = F.l1_loss(pred_amp, true_amp)
        
        # Phase loss (complex difference)
        phase_loss = torch.mean(torch.abs(pred_fft - true_fft))
        
        # Combined loss
        total_loss = self.amp_weight * amp_loss + self.phase_weight * phase_loss
        
        return total_loss


class AdaptiveFrequencyLoss(nn.Module):
    """
    Adaptive Frequency Loss with group-specific lambda weights
    """
    
    def __init__(self, lambda_weights=None):
        super(AdaptiveFrequencyLoss, self).__init__()
        
        if lambda_weights is None:
            # Default lambda weights for each group
            lambda_weights = {
                'HF': 0.05,
                'Mid': 0.2,
                'Seasonal': 0.8,
                'Trend': 0.3
            }
        
        self.lambda_weights = lambda_weights
        self.freq_loss = FrequencyLoss()
        
    def forward(self, pred, true, group_name):
        """
        Compute frequency loss with group-specific weight
        
        Args:
            pred: Predicted signal
            true: Ground truth signal
            group_name: Name of IMF group ('HF', 'Mid', 'Seasonal', 'Trend')
            
        Returns:
            Weighted frequency loss
        """
        lambda_val = self.lambda_weights.get(group_name, 0.2)
        freq_loss_val = self.freq_loss(pred, true)
        
        return lambda_val * freq_loss_val


# ==================== Combined Loss ====================

class HybridLoss(nn.Module):
    """
    Combined Time-Domain + Frequency-Domain Loss
    L = L_time + λ * L_freq
    """
    
    def __init__(self, time_loss='mse', lambda_weights=None, 
                 amp_weight=1.0, phase_weight=0.5):
        super(HybridLoss, self).__init__()
        
        # Time loss
        if time_loss == 'mse':
            self.time_loss_fn = nn.MSELoss()
        elif time_loss == 'mae':
            self.time_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown time loss: {time_loss}")
        
        # Frequency loss
        self.freq_loss = AdaptiveFrequencyLoss(lambda_weights)
        
    def forward(self, pred, true, group_name='Mid'):
        """
        Compute hybrid loss
        
        Args:
            pred: Predicted signal [batch, seq_len, 1]
            true: Ground truth signal [batch, seq_len, 1]
            group_name: IMF group name for adaptive lambda
            
        Returns:
            total_loss, time_loss, freq_loss (for logging)
        """
        # Time-domain loss
        time_loss = self.time_loss_fn(pred, true)
        
        # Frequency-domain loss (with group-specific lambda)
        freq_loss = self.freq_loss(pred, true, group_name)
        
        # Total loss
        total_loss = time_loss + freq_loss
        
        return total_loss, time_loss, freq_loss


# ==================== Helper Functions ====================

def get_fusion_module(fusion_type='weighted', num_groups=4, d_model=64):
    """
    Factory function for fusion modules
    
    Args:
        fusion_type: 'weighted' or 'gated'
        num_groups: Number of IMF groups
        d_model: Embedding dimension for gated fusion
        
    Returns:
        Fusion module instance
    """
    if fusion_type == 'weighted':
        return WeightedFusion(num_groups=num_groups)
    elif fusion_type == 'gated':
        return GatedFusion(num_groups=num_groups, d_model=d_model)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def spectral_analysis(signal):
    """
    Perform spectral analysis on a signal
    Useful for debugging frequency loss
    
    Args:
        signal: [batch, seq_len] tensor
        
    Returns:
        Dictionary with spectral information
    """
    # FFT
    fft_vals = torch.fft.rfft(signal, dim=-1)
    
    # Amplitude spectrum
    amplitude = torch.abs(fft_vals)
    
    # Phase spectrum
    phase = torch.angle(fft_vals)
    
    # Power spectrum
    power = amplitude ** 2
    
    return {
        'amplitude': amplitude,
        'phase': phase,
        'power': power,
        'fft_complex': fft_vals
    }


# ==================== Testing ====================

if __name__ == "__main__":
    print("Testing Fusion and Frequency Loss Modules")
    print("=" * 70)
    
    # Setup
    batch_size = 8
    pred_len = 24
    num_groups = 4
    
    # Dummy group predictions
    group_preds = [
        torch.randn(batch_size, pred_len, 1) for _ in range(num_groups)
    ]
    
    # Test Weighted Fusion
    print("\n1. Testing Weighted Fusion:")
    weighted_fusion = WeightedFusion(num_groups=num_groups)
    fused_weighted = weighted_fusion(group_preds)
    print(f"   Output shape: {fused_weighted.shape}")
    print(f"   Learned weights: {F.softmax(weighted_fusion.weights, dim=0)}")
    
    # Test Gated Fusion
    print("\n2. Testing Gated Fusion:")
    gated_fusion = GatedFusion(num_groups=num_groups, d_model=64)
    fused_gated = gated_fusion(group_preds)
    print(f"   Output shape: {fused_gated.shape}")
    
    # Test Frequency Loss
    print("\n3. Testing Frequency Loss:")
    freq_loss_fn = FrequencyLoss()
    pred = torch.randn(batch_size, pred_len, 1)
    true = torch.randn(batch_size, pred_len, 1)
    freq_loss = freq_loss_fn(pred, true)
    print(f"   Frequency loss: {freq_loss.item():.6f}")
    
    # Test Adaptive Frequency Loss
    print("\n4. Testing Adaptive Frequency Loss:")
    adaptive_freq_loss = AdaptiveFrequencyLoss()
    for group in ['HF', 'Mid', 'Seasonal', 'Trend']:
        loss = adaptive_freq_loss(pred, true, group)
        lambda_val = adaptive_freq_loss.lambda_weights[group]
        print(f"   {group:10s}: λ={lambda_val:.2f}, loss={loss.item():.6f}")
    
    # Test Hybrid Loss
    print("\n5. Testing Hybrid Loss:")
    hybrid_loss = HybridLoss(time_loss='mse')
    total, time_l, freq_l = hybrid_loss(pred, true, group_name='Seasonal')
    print(f"   Total loss: {total.item():.6f}")
    print(f"   Time loss:  {time_l.item():.6f}")
    print(f"   Freq loss:  {freq_l.item():.6f}")
    
    # Test Spectral Analysis
    print("\n6. Testing Spectral Analysis:")
    signal = torch.randn(batch_size, 100)
    spectra = spectral_analysis(signal)
    print(f"   Amplitude spectrum shape: {spectra['amplitude'].shape}")
    print(f"   Phase spectrum shape:     {spectra['phase'].shape}")
    print(f"   Power spectrum shape:     {spectra['power'].shape}")
    
    print("\n" + "=" * 70)