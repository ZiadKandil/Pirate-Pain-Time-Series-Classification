import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PiratePainDataset(Dataset):

    # initialize the dataset from CSV files and configuration
    def __init__(self, data_path, labels_pth, config, mode='train'):
        
        self.data_path = data_path
        self.labels_path = labels_pth
        self.config = config
        self.mode = mode
        self.seq_length = config['data']['seq_length']

        # Load data and labels
        self.data, self.labels, self.sample_indices = self._load_data()

        # Preprocess the data
        self._preprocess()

    # Function to load the data from CSV and reshape it into sequences and load labels and encode them
    def _load_data(self):
        
        df = pd.read_csv(self.data_path)

        # Unique sample indices
        sample_indices = df['sample_index'].unique()

        # Feature columns (exclude sample_index and time)
        feature_cols = []
        for col in df.columns:
            if col not in ['sample_index', 'time']:
                feature_cols.append(col)
        
        n_samples = len(sample_indices)
        n_features = len(feature_cols)

        # Reshape to sequences
        sequences = np.zeros((n_samples, self.seq_length, n_features))
        for i, sample_idx in enumerate(sample_indices):
            # Extract the data for this sample index and put it in the sequences array
            sample_data = df[df['sample_index'] == sample_idx][feature_cols].values
            sequences[i] = sample_data[:self.seq_length]
        
        # Load labels 
        labels = None
        if self.labels_path is not None:
            labels_df = pd.read_csv(self.labels_path)
            labels = labels_df['label'].values
            # Encode labels
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(labels)
        
        return sequences, labels, sample_indices
    
    # Function to preprocess the data by normalizing it, handle missing values, augment it and finally splitting it into train and validation sets
    def _preprocess(self):

        # Normalize the data using StandardScaler
        if self.config['data']['normalize'] == True:
            n_samples, seq_len, n_features = self.data.shape

            # Reshape for scaling by flattening time steps
            data_reshaped = self.data.reshape(-1, n_features)

            # Fit the scaler on the training data and transform on val/test
            if self.mode == 'train':
                self._scalar = StandardScaler()
                data_scaled = self._scalar.fit_transform(data_reshaped)
            else:
                data_scaled = self._scalar.transform(data_reshaped)

            # Reshape back to original shape
            self.data = data_scaled.reshape(n_samples, seq_len, n_features)
        
        # Handle missing values with the specified strategy
        if self.config['data']['Missing_value_strategy'] == 'mean':
            self.data = np.nan_to_num(self.data, nan = np.nanmean(self.data))

        elif self.config['data']['Missing_value_strategy'] == 'forward_fill':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[2]):
                    series = self.data[i, :, j]
                    if np.isnan(series).any():
                        df = pd.Series(series)
                        df = df.ffill().bfill()  # Forward fill followed by backward fill to handle any remaining NaNs
                        self.data[i, :, j] = df.values

        elif self.config['data']['Missing_value_strategy'] == 'backward_fill':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[2]):
                    series = self.data[i, :, j]
                    if np.isnan(series).any():
                        df = pd.Series(series)
                        df = df.bfill().ffill() 
                        self.data[i, :, j] = df.values
        
        # Data augmentation by adding Gaussian noise if specified in the config
        if self.config['data']['augment'] == True:
            self.noise_level = self.config['data']['Augmentation_noise_level']
            noise = np.random.normal(0, self.noise_level, self.data.shape)
            augmented_data = self.data + noise
            self.data = np.concatenate([self.data, augmented_data], axis=0)
            self.labels = np.concatenate([self.labels, self.labels], axis=0)

        # Split into train and validation sets if in training mode
        if self.mode in ['train', 'val'] and self.labels is not None:
            self.val_split = self.config['data'].get('val_split', 0.2)
            indices = np.arange(len(self.data))
            train_idx, val_idx = train_test_split(
                indices,
                test_size = self.val_split,
                stratify = self.labels,  # Stratify to maintain label distribution in train and val sets
                random_state = self.config['training']['seed']  # Set random seed for reproducibility
            )
            if self.mode == 'train':
                self.data = self.data[train_idx]
                self.labels = self.labels[train_idx]
            else: 
                self.data = self.data[val_idx]
                self.labels = self.labels[val_idx]
    
    # Function to return the length of the dataset
    def __len__(self):
        return len(self.data)
    
    # Function to get a sample from the dataset
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.data[idx])
        if self.labels is not None:
            label = torch.LongTensor(self.labels[idx])[0]
            return sequence, label
        else:
            return sequence
        