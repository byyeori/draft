"""
IMF Feature Extraction and Automatic Grouping Module
Analyzes IMF characteristics and groups them for model selection
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml


class IMFAnalyzer:
    """
    Extracts features from IMFs and performs automatic grouping
    """
    
    def __init__(self, sampling_rate=1.0):
        """
        Args:
            sampling_rate: Sampling rate of the signal (Hz)
        """
        self.sampling_rate = sampling_rate
        self.features = []
        self.groups = {}
        self.scaler = None
        self.kmeans = None
        
    def extract_features(self, imfs):
        """
        Extract features from all IMFs
        
        Args:
            imfs: List of IMF arrays
            
        Returns:
            List of feature dictionaries
        """
        self.features = []
        
        for i, imf in enumerate(imfs):
            features = {
                'imf_index': i,
                'dominant_freq': self._compute_dominant_freq(imf),
                'spectral_entropy': self._compute_spectral_entropy(imf),
                'zcr': self._compute_zcr(imf),
                'periodicity': self._compute_periodicity(imf),
                'trend_strength': self._compute_trend_strength(imf)
            }
            self.features.append(features)
        
        return self.features
    
    def _compute_dominant_freq(self, imf):
        """Compute dominant frequency using FFT"""
        n = len(imf)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        imf_windowed = imf * window
        
        # FFT
        fft_vals = fft(imf_windowed)
        freqs = fftfreq(n, 1.0 / self.sampling_rate)
        
        # Only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        # Find peak
        if len(fft_vals) > 0:
            peak_idx = np.argmax(fft_vals)
            dominant_freq = freqs[peak_idx]
        else:
            dominant_freq = 0.0
        
        return dominant_freq
    
    def _compute_spectral_entropy(self, imf):
        """Compute spectral entropy (frequency distribution spread)"""
        n = len(imf)
        
        # FFT
        fft_vals = np.abs(fft(imf))[:n // 2]
        
        # Normalize to probability distribution
        fft_vals = fft_vals / (np.sum(fft_vals) + 1e-10)
        
        # Compute entropy
        spec_entropy = entropy(fft_vals + 1e-10)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(fft_vals))
        normalized_entropy = spec_entropy / (max_entropy + 1e-10)
        
        return normalized_entropy
    
    def _compute_zcr(self, imf):
        """Compute zero-crossing rate"""
        # Count sign changes
        sign_changes = np.sum(np.abs(np.diff(np.sign(imf))) > 0)
        
        # Normalize by length
        zcr = sign_changes / (len(imf) - 1)
        
        return zcr
    
    def _compute_periodicity(self, imf):
        """Compute periodicity strength using autocorrelation"""
        # Autocorrelation
        autocorr = np.correlate(imf - np.mean(imf), 
                               imf - np.mean(imf), 
                               mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Normalize
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Find peaks in autocorrelation (excluding lag 0)
        if len(autocorr) > 10:
            peaks, _ = sp_signal.find_peaks(autocorr[1:], height=0.2)
            
            if len(peaks) > 0:
                # Periodicity strength is max peak height
                periodicity = np.max(autocorr[peaks + 1])
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0
        
        return periodicity
    
    def _compute_trend_strength(self, imf):
        """Compute trend strength using linear regression R²"""
        n = len(imf)
        x = np.arange(n)
        
        # Linear regression
        coeffs = np.polyfit(x, imf, 1)
        trend = np.polyval(coeffs, x)
        
        # R² score
        ss_res = np.sum((imf - trend) ** 2)
        ss_tot = np.sum((imf - np.mean(imf)) ** 2)
        
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        r_squared = max(0, r_squared)  # Ensure non-negative
        
        return r_squared
    
    def group_imfs_rule_based(self, config_path='utils/grouping_config.yaml'):
        """
        Group IMFs using rule-based approach
        
        Args:
            config_path: Path to grouping configuration file
            
        Returns:
            Dictionary mapping group names to IMF indices
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rules = config['rule_based']
        
        groups = {
            'HF': [],
            'Mid': [],
            'Seasonal': [],
            'Trend': []
        }
        
        for feat in self.features:
            idx = feat['imf_index']
            
            # Check if it's trend (last IMF usually)
            if idx == len(self.features) - 1:
                groups['Trend'].append(idx)
            
            # Check HF rules
            elif (feat['zcr'] >= rules['hf']['zcr_min'] and
                  feat['spectral_entropy'] >= rules['hf']['entropy_min'] and
                  feat['periodicity'] <= rules['hf']['periodicity_max']):
                groups['HF'].append(idx)
            
            # Check Seasonal rules
            elif feat['periodicity'] >= rules['seasonal']['periodicity_min']:
                groups['Seasonal'].append(idx)
            
            # Check Trend rules (if not last IMF)
            elif (feat['trend_strength'] >= rules['trend']['trend_strength_min'] and
                  feat['zcr'] <= rules['trend']['zcr_max']):
                groups['Trend'].append(idx)
            
            # Default to Mid
            else:
                groups['Mid'].append(idx)
        
        self.groups = groups
        return groups
    
    def group_imfs_clustering(self, n_clusters=4, fit=True, 
                             config_path='utils/grouping_config.yaml'):
        """
        Group IMFs using K-Means clustering
        
        IMPORTANT: Use fit=True ONLY for training data
                  Use fit=False for validation/test to prevent data leakage
        
        Args:
            n_clusters: Number of clusters
            fit: Whether to fit the clustering model (True for train only)
            config_path: Path to configuration file
            
        Returns:
            Dictionary mapping group names to IMF indices
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        feature_names = config['kmeans']['features']
        
        # Extract feature matrix
        X = []
        for feat in self.features:
            x = [feat[name] for name in feature_names]
            X.append(x)
        X = np.array(X)
        
        if fit:
            # FIT ONLY ON TRAINING DATA
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=config['kmeans']['random_state'],
                n_init=10
            )
            labels = self.kmeans.fit_predict(X_scaled)
        else:
            # PREDICT ONLY FOR VALID/TEST DATA
            if self.scaler is None or self.kmeans is None:
                raise ValueError("Must fit clustering on training data first!")
            
            X_scaled = self.scaler.transform(X)
            labels = self.kmeans.predict(X_scaled)
        
        # Map clusters to semantic groups based on centroids
        groups = self._map_clusters_to_groups(labels, X_scaled)
        
        self.groups = groups
        return groups
    
    def _map_clusters_to_groups(self, labels, X_scaled):
        """
        Map cluster labels to semantic group names
        Based on cluster characteristics
        """
        unique_labels = np.unique(labels)
        
        # Compute cluster characteristics
        cluster_chars = {}
        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            
            # Average features for this cluster
            avg_zcr = np.mean([self.features[i]['zcr'] for i in indices])
            avg_periodicity = np.mean([self.features[i]['periodicity'] for i in indices])
            avg_trend = np.mean([self.features[i]['trend_strength'] for i in indices])
            avg_entropy = np.mean([self.features[i]['spectral_entropy'] for i in indices])
            
            cluster_chars[label] = {
                'indices': indices.tolist(),
                'zcr': avg_zcr,
                'periodicity': avg_periodicity,
                'trend': avg_trend,
                'entropy': avg_entropy
            }
        
        # Assign semantic names based on characteristics
        groups = {'HF': [], 'Mid': [], 'Seasonal': [], 'Trend': []}
        
        assigned = set()
        
        # Assign HF: high ZCR + high entropy
        for label, chars in cluster_chars.items():
            if chars['zcr'] > 0.3 and chars['entropy'] > 0.6:
                groups['HF'].extend(chars['indices'])
                assigned.add(label)
        
        # Assign Seasonal: high periodicity
        for label, chars in cluster_chars.items():
            if label not in assigned and chars['periodicity'] > 0.5:
                groups['Seasonal'].extend(chars['indices'])
                assigned.add(label)
        
        # Assign Trend: high trend strength + low ZCR
        for label, chars in cluster_chars.items():
            if label not in assigned and chars['trend'] > 0.5:
                groups['Trend'].extend(chars['indices'])
                assigned.add(label)
        
        # Remaining goes to Mid
        for label, chars in cluster_chars.items():
            if label not in assigned:
                groups['Mid'].extend(chars['indices'])
        
        return groups
    
    def get_group_for_imf(self, imf_index):
        """Get group name for specific IMF index"""
        for group_name, indices in self.groups.items():
            if imf_index in indices:
                return group_name
        return 'Mid'  # Default
    
    def print_feature_summary(self):
        """Print summary of extracted features"""
        if not self.features:
            print("No features extracted yet.")
            return
        
        print("\n" + "=" * 80)
        print("IMF Feature Summary")
        print("=" * 80)
        print(f"{'IMF':<5} {'Dom.Freq':<12} {'Entropy':<10} {'ZCR':<10} "
              f"{'Period.':<10} {'Trend':<10}")
        print("-" * 80)
        
        for feat in self.features:
            print(f"{feat['imf_index']:<5} "
                  f"{feat['dominant_freq']:<12.4f} "
                  f"{feat['spectral_entropy']:<10.4f} "
                  f"{feat['zcr']:<10.4f} "
                  f"{feat['periodicity']:<10.4f} "
                  f"{feat['trend_strength']:<10.4f}")
        print("=" * 80)
    
    def print_group_summary(self):
        """Print summary of IMF groups"""
        if not self.groups:
            print("No groups defined yet.")
            return
        
        print("\n" + "=" * 60)
        print("IMF Grouping Summary")
        print("=" * 60)
        
        for group_name, indices in self.groups.items():
            print(f"{group_name:<12}: IMFs {indices}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("IMF Analyzer Example")
    
    # Load example IMFs
    imfs = np.load('data/imfs/example_imfs.npy')
    imfs = [imfs[i] for i in range(len(imfs))]
    
    # Initialize analyzer
    analyzer = IMFAnalyzer(sampling_rate=100.0)
    
    # Extract features
    features = analyzer.extract_features(imfs)
    analyzer.print_feature_summary()
    
    # Rule-based grouping
    print("\n--- Rule-Based Grouping ---")
    groups_rule = analyzer.group_imfs_rule_based()
    analyzer.print_group_summary()
    
    # K-Means grouping (fit on train)
    print("\n--- K-Means Clustering Grouping ---")
    groups_kmeans = analyzer.group_imfs_clustering(n_clusters=4, fit=True)
    analyzer.print_group_summary()