"""
Prediction script for Pirate Pain Challenge
"""
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PiratePainDataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """Get model based on configuration"""
    model_type = config['model']['type']
    
    if model_type == 'gru':
        from models.gru.gru_model import GRUModel
        model = GRUModel(config)
    elif model_type == 'lstm':
        from models.lstm.lstm_model import LSTMModel
        model = LSTMModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def predict(model, data_loader, device):
    """Make predictions"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Predicting'):
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


def main():
    parser = argparse.ArgumentParser(description='Predict with Pirate Pain Classification Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_dataset = PiratePainDataset(
        data_path=config['data']['test_path'],
        labels_path=None,
        config=config,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict(model, test_loader, device)
    
    # Convert predictions to labels
    label_mapping = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
    predicted_labels = [label_mapping[p] for p in predictions]
    
    # Create submission file
    submission = pd.DataFrame({
        'sample_index': range(len(predictions)),
        'pain_level': predicted_labels
    })
    
    submission.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"\nPrediction distribution:")
    print(submission['pain_level'].value_counts())


if __name__ == '__main__':
    main()
