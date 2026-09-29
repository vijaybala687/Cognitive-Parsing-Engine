import numpy as np
from scipy.signal import welch

class FatigueMonitor:
    def __init__(self, fs=160):
        # fs is your sampling rate (PhysioNet uses 160Hz)
        self.fs = fs
        self.theta_band = (4, 8)
        self.alpha_band = (8, 12)

    def calculate_fatigue_ratio(self, eeg_epoch):
        """
        Takes a raw 4-second EEG array (e.g., shape: [64 channels, 640 samples])
        Returns the Theta/Alpha ratio. A higher ratio indicates severe mental fatigue.
        """
        # Calculate Power Spectral Density using Welch's method
        freqs, psd = welch(eeg_epoch, fs=self.fs, nperseg=self.fs*2, axis=-1)
        
        # Find intersecting indices for Theta (4-8 Hz) and Alpha (8-12 Hz)
        theta_idx = np.logical_and(freqs >= self.theta_band[0], freqs <= self.theta_band[1])
        alpha_idx = np.logical_and(freqs >= self.alpha_band[0], freqs <= self.alpha_band[1])
        
        # Calculate average power in each band across all motor channels
        theta_power = np.mean(psd[:, theta_idx])
        alpha_power = np.mean(psd[:, alpha_idx])
        
        # Prevent division by zero
        if alpha_power == 0:
            return 1.0 
            
        fatigue_ratio = theta_power / alpha_power
        return fatigue_ratio

    def get_difficulty_level(self, fatigue_ratio, baseline_stroke_level):
        """
        Determines the game's difficulty (1-3) based on real-time exhaustion.
        """
        # Example threshold logic - will need empirical tuning
        if fatigue_ratio > 1.5 or baseline_stroke_level == "severe":
            return 1  # Maximum assistance, binary classification only
        elif fatigue_ratio > 1.0:
            return 2  # Moderate assistance
        else:
            return 3  # Zero assistance, full 4-class control