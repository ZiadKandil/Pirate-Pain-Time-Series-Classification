"""
Preprocessing utilities
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_sequences(sequences):
    """
    Normalize sequences using StandardScaler
    
    Args:
        sequences: Array of shape (n_samples, seq_length, n_features)
    
    Returns:
        Normalized sequences
    """
    n_samples, seq_length, n_features = sequences.shape
    
    # Reshape to 2D
    sequences_reshaped = sequences.reshape(-1, n_features)
    
    # Scale
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    
    # Reshape back to 3D
    sequences_scaled = sequences_scaled.reshape(n_samples, seq_length, n_features)
    
    return sequences_scaled, scaler


def handle_missing_values(sequences, strategy='forward_fill'):
    """
    Handle missing values in sequences
    
    Args:
        sequences: Array of shape (n_samples, seq_length, n_features)
        strategy: 'forward_fill', 'backward_fill', or 'mean'
    
    Returns:
        Sequences with missing values handled
    """
    if strategy == 'forward_fill':
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[2]):
                series = sequences[i, :, j]
                if np.isnan(series).any():
                    df = pd.Series(series)
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    sequences[i, :, j] = df.values
    
    elif strategy == 'backward_fill':
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[2]):
                series = sequences[i, :, j]
                if np.isnan(series).any():
                    df = pd.Series(series)
                    df = df.fillna(method='bfill').fillna(method='ffill')
                    sequences[i, :, j] = df.values
    
    elif strategy == 'mean':
        sequences = np.nan_to_num(sequences, nan=np.nanmean(sequences))
    
    return sequences


def augment_sequences(sequences, labels, noise_level=0.01):
    """
    Augment sequences with Gaussian noise
    
    Args:
        sequences: Array of shape (n_samples, seq_length, n_features)
        labels: Array of labels
        noise_level: Standard deviation of noise
    
    Returns:
        Augmented sequences and labels
    """
    noise = np.random.normal(0, noise_level, sequences.shape)
    sequences_aug = sequences + noise
    
    sequences_combined = np.concatenate([sequences, sequences_aug], axis=0)
    labels_combined = np.concatenate([labels, labels], axis=0)
    
    return sequences_combined, labels_combined
