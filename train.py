import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm
from sklearn.utils.class_weight import compute_class_weight

from utils.dataset import PiratePainDataset
from utils.metrics import calculate_metrics

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

# train for one epoch
def train_epoch(model, train_loader, criterion, optimizer, device):

    model.train() # Set the model to training mode

    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    pbar = tqdm.tqdm(train_loader, desc='Training')

    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)  # Move data and target tensors to the specified device
        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        output = model(data)  # Forward pass: compute the output of the model given the input data
        loss = criterion(output, target)  # Calculate the loss between output and target
        loss.backward()   # Backpropagate the loss to compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip the gradients to prevent exploding gradients
        optimizer.step()  # Update the model parameters based on the computed gradients

        total_loss += loss.item()  
        pred = output.argmax(dim=1)  # Get the index of the max probability as the predicted class

        correct += pred.eq(target).sum().item()
        total += target.size(0)

        all_targets.extend(target.cpu().numpy())  # Collect all target labels for metrics calculation
        all_preds.extend(pred.cpu().numpy())   # Collect all predicted labels for metrics calculation

        pbar.set_postfix({'loss': total_loss / (batch_idx + 1), 'accuracy': 100.*correct / total})
    
    metrics = calculate_metrics(all_targets, all_preds)  # Calculate metrics for the epoch
    f1_score = metrics['f1_macro']

    return total_loss / len(train_loader), f1_score

# validate for one epoch
def validate_epoch(model, val_loader, criterion, device):

    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_targets = []
    all_preds = []

    pbar = tqdm.tqdm(val_loader, desc='Validation')

    with torch.no_grad(): # Disable gradient calculation for validation
        for data, target in pbar:

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    metrics = calculate_metrics(all_targets, all_preds)
    f1_score = metrics['f1_macro']

    return total_loss / len(val_loader), f1_score

def main():

    # Create a parser oobject to handle command-line arguments
    parser = argparse.ArgumentParser(description='Train Pirate Pain Classification Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    args = parser.parse_args()  # Parse the command-line arguments

    # Load the configuration from the specified config file
    config = load_config(args.config)
    print('Configuration loaded from', args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # set random seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # load and preprocess the dataset
    print('Loading data...')

    train_dataset = PiratePainDataset(
        data_path=config['data']['train_path'],
        labels_path=config['data']['labels_path'],
        config=config,
        mode='train'
    )

    val_dataset = PiratePainDataset(
        data_path=config['data']['train_path'],
        labels_path=config['data']['labels_path'],
        config=config,
        mode='val',
        scaler=train_dataset._scalar
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config['training']['batch_size'],
        shuffle = True,
        num_workers = config['training'].get('num_workers', 4)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config['training']['batch_size'],
        shuffle = False,
        num_workers = config['training'].get('num_workers', 4)
    )

    print(f"Train samples: {len(train_dataset)}, 'Val samples: {len(val_dataset)}")

    # Calculating class weights to consider class imbalance
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_dataset.labels),
        y = train_dataset.labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Create model
    print('Creating model...')

    model = get_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    # Scheduler for learning rate decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    start_epoch = 0
    best_val_f1 = 0.0

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)

    patience_counter = 0
    patience = config['training'].get('early_stopping_patience', 10)

    # Training loop
    print('\nStarting training...')

    for epoch in range(start_epoch, config['training']['num_epochs']):

        print(f'\nEpoch {epoch+1}/{config['training']['num_epochs']}')

        # Train
        train_loss, train_f1 = train_epoch(model, train_loader,
                                            criterion, optimizer, device)
        
        # Validate
        val_loss, val_f1 = validate_epoch(model, val_loader,
                                            criterion, device)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # Save the best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            save_path = os.path.join('checkpoints', f'{config['model']['type']}_best.pth')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'config': config
            }, save_path)

            print(f"Saved best model to {save_path}")

        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    print(f'\nTraining completed, Best validation F1 score: {best_val_f1:.4f}')

if __name__ == '__main__':
    main()
