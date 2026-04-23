import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.dataset import PiratePainDataset

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Get model from configuartion
def get_model(config):
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

# Make predictions
def predict(model, data_loader, device):

    model.eval()
    predictions = []
    probabilities = []

    pbar = tqdm.tqdm(data_loader, desc='Predicting')

    with torch.no_grad():
        for data in enumerate(pbar):
            # Check if data is a tuple (data, target) or just data
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
    parser.add_argument('--output', type=str, default='submission.csv', help='Path to save predictions')
    args = parser.parse_args()

    config = load_config(args.config)
    print('Configuration loaded from', args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print('Looading test data...')

    test_dataset = PiratePainDataset(
        data_path=config['data']['test_path'],
        labels_path=None,
        config=config,
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle= False,
        num_workers=config['training'].get('num_workers', 4)
    )

    print(f'Test Samples: {len(test_dataset)}')

    print(f'Loading model from {args.checkpoint}')
    
    model.get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print('Predicting...')

    predictions, probabilities = predict(model, test_loader, device)

    # Convert predictions to original labels if label encoder is available
    label_mapping = {0: 'high_pain', 1: 'low_pain', 2: 'no_pain'}
    predicted_labels = [label_mapping[p] for p in predictions]

    # Save predictions to CSV
    submission = pd.DataFrame({
        'sample_index': range(len(predictions)),
        'pain_level': predicted_labels
    })

    submission.to_csv(arg.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"\nPrediction distribution:")
    print(submission['pain_level'].value_counts())


if __name__ == '__main__':
    main()
