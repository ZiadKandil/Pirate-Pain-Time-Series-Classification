"""
Dataset utilities for Pirate Pain Challenge
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PiratePainDataset(Dataset):
    """Dataset for pirate pain time series classification"""
    
    # Class-level scaler to share between train/val
    _scaler = None
    
    def __init__(self, data_path, labels_path, config, mode='train'):
        """
        Args:
            data_path: Path to CSV file with time series data
            labels_path: Path to CSV file with labels
            config: Configuration dictionary
            mode: 'train', 'val', or 'test'
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.config = config
        self.mode = mode
        self.seq_length = config['data']['seq_length']
        
        # Load data
        self.data, self.labels, self.sample_indices = self._load_data()
        
        # Preprocess
        if mode in ['train', 'val', 'test']:
            self._preprocess()
    
    def _load_data(self):
        """Load and reshape data"""
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Get unique sample indices
        sample_indices = df['sample_index'].unique()
        
        # Get feature columns (exclude sample_index and time)
        feature_cols = [col for col in df.columns if col not in ['sample_index', 'time']]
        
        # Reshape to sequences
        n_samples = len(sample_indices)
        n_features = len(feature_cols)
        sequences = np.zeros((n_samples, self.seq_length, n_features))
        
        for i, sample_idx in enumerate(sample_indices):
            sample_data = df[df['sample_index'] == sample_idx][feature_cols].values
            sequences[i] = sample_data[:self.seq_length]
        
        # Load labels if available
        labels = None
        if self.labels_path is not None:
            labels_df = pd.read_csv(self.labels_path)
            labels = labels_df['pain_level'].values
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(labels)
        
        return sequences, labels, sample_indices
    
    def _preprocess(self):
        """Preprocess data (normalize)"""
        if self.config['data'].get('normalize', True):
            n_samples, seq_len, n_features = self.data.shape
            
            # Reshape for scaling
            data_reshaped = self.data.reshape(-1, n_features)
            
            # Scale - fit on train, transform on val/test
            if self.mode == 'train':
                if PiratePainDataset._scaler is None:
                    PiratePainDataset._scaler = StandardScaler()
                data_scaled = PiratePainDataset._scaler.fit_transform(data_reshaped)
            else:
                # Use fitted scaler for val/test
                if PiratePainDataset._scaler is None:
                    # Fallback: fit if not already fitted
                    PiratePainDataset._scaler = StandardScaler()
                    data_scaled = PiratePainDataset._scaler.fit_transform(data_reshaped)
                else:
                    data_scaled = PiratePainDataset._scaler.transform(data_reshaped)
            
            # Reshape back
            self.data = data_scaled.reshape(n_samples, seq_len, n_features)
        
        # Split train/val
        if self.mode in ['train', 'val'] and self.labels is not None:
            val_split = self.config['data'].get('val_split', 0.2)
            
            indices = np.arange(len(self.data))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_split,
                stratify=self.labels,
                random_state=self.config['training']['seed']
            )
            
            if self.mode == 'train':
                self.data = self.data[train_idx]
                self.labels = self.labels[train_idx]
            else:  # val
                self.data = self.data[val_idx]
                self.labels = self.labels[val_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item"""
        sequence = torch.FloatTensor(self.data[idx])
        
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])[0]
            return sequence, label
        else:
            return sequence
