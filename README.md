# 🏴‍☠️ Pirate Pain Challenge - Time Series Classification

Deep learning models for multivariate time series classification to predict pain levels from temporal body movement data.

## 📋 Problem Description

This challenge involves classifying multivariate time series data captured from subjects over repeated observations. Each sample contains temporal dynamics of body joints and pain perception measurements.

**Task**: Predict the subject's true pain status from 180 timestep sequences.

**Classes**:
- `no_pain` - No pain detected
- `low_pain` - Low pain level  
- `high_pain` - High pain level

## 📊 Dataset Overview

Each record represents a time step within a subject's recording:

### Features (38 total):
- **Pain Surveys** (4): `pain_survey_1` to `pain_survey_4` - Rule-based sensor aggregations estimating perceived pain
- **Subject Characteristics** (3): `n_legs`, `n_hands`, `n_eyes` - Physical attributes
- **Joint Measurements** (31): `joint_00` to `joint_30` - Continuous measurements of body joint angles (neck, elbow, knee, etc.) over time

### Data Files:
- `pirate_pain_train.csv` - Training sequences (180 timesteps × 38 features per sample)
- `pirate_pain_train_labels.csv` - Training labels (pain_level per sample)
- `pirate_pain_test.csv` - Test sequences for prediction
- `sample_submission.csv` - Example submission format

## 🗂️ Project Structure

```
.
├── train.py              # Main training script
├── predict.py            # Prediction and submission script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
│
├── configs/             # YAML configuration files
│   ├── gru_config.yaml
│   └── lstm_config.yaml
│
├── data/                # Dataset location (configure paths in YAML)
│   ├── train_data/      # Reserved for processed data
│   └── test_data/       # Reserved for processed data
│
├── models/              # Neural network architectures
│   ├── gru/
│   │   ├── gru_model.py
│   │   └── __init__.py
│   └── lstm/
│       ├── lstm_model.py
│       └── __init__.py
│
└── utils/               # Utility modules
    ├── dataset.py       # Dataset class with preprocessing
    ├── metrics.py       # Evaluation metrics
    ├── preprocessing.py # Data preprocessing utilities
    └── __init__.py
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

**Required packages**: torch, numpy, pandas, scikit-learn, pyyaml, tqdm, matplotlib, seaborn

### 2. Configure Data Paths

Edit the config file (e.g., `configs/gru_config.yaml`) to point to your data:

```yaml
data:
  train_path: ../Data/pirate_pain_train.csv
  labels_path: ../Data/pirate_pain_train_labels.csv
  test_path: ../Data/pirate_pain_test.csv
```

### 3. Train a Model

Train with GRU:
```bash
python train.py --config configs/gru_config.yaml
```

Train with LSTM:
```bash
python train.py --config configs/lstm_config.yaml
```

**Training features**:
- Automatic train/validation split (80/20)
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Checkpoint saving

### 4. Make Predictions

```bash
python predict.py --config configs/gru_config.yaml --checkpoint checkpoints/gru_best.pth --output submission.csv
```

This creates a CSV file with predictions in the format:
```
sample_index,pain_level
0,no_pain
1,low_pain
2,high_pain
...
```

## 🧠 Available Models

### 1. **GRU Model** (Recommended)
- **Architecture**: Bidirectional GRU with 2 layers
- **Hidden Size**: 128
- **Dropout**: 0.3
- **Strengths**: Fast training, good performance on sequential data
- **Config**: `configs/gru_config.yaml`

### 2. **LSTM Model**
- **Architecture**: Bidirectional LSTM with 2 layers
- **Hidden Size**: 128
- **Dropout**: 0.3
- **Strengths**: Handles long-term dependencies well
- **Config**: `configs/lstm_config.yaml`

Both models use:
- Batch normalization
- Dropout for regularization
- Adam optimizer with weight decay
- Cross-entropy loss

## ⚙️ Configuration

Edit YAML files in `configs/` to customize:

```yaml
model:
  type: gru                    # Model type: 'gru' or 'lstm'
  input_size: 38               # Number of features
  hidden_size: 128             # Hidden layer size
  num_layers: 2                # Number of recurrent layers
  num_classes: 3               # Output classes
  dropout: 0.3                 # Dropout probability
  bidirectional: true          # Use bidirectional RNN

training:
  batch_size: 32               # Batch size
  num_epochs: 100              # Maximum epochs
  learning_rate: 0.001         # Learning rate
  weight_decay: 0.0001         # L2 regularization
  early_stopping_patience: 10  # Early stopping patience
  num_workers: 0               # DataLoader workers (0 for Windows)
  seed: 42                     # Random seed

data:
  seq_length: 180              # Sequence length
  val_split: 0.2               # Validation split ratio
  normalize: true              # Apply StandardScaler normalization
```

## 📈 Training Details

### Preprocessing Pipeline:
1. Load CSV data with `sample_index` and `time` columns
2. Reshape flat data to 3D sequences: `(n_samples, 180, 38)`
3. Apply StandardScaler normalization (fit on train, transform on val/test)
4. Encode labels: `no_pain→0, low_pain→1, high_pain→2`
5. Split into train/validation sets (stratified)

### Training Process:
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Cross-entropy
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Patience of 10 epochs
- **Checkpointing**: Saves best model based on validation accuracy

### Monitoring:
- Training/validation loss and accuracy printed each epoch
- Progress bars with tqdm
- Best model automatically saved to `checkpoints/`

## 📊 Evaluation

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro and weighted averages
- **Confusion Matrix**: Per-class performance

Evaluation metrics are calculated in `utils/metrics.py`.

## 💡 Tips for Better Performance

1. **Hyperparameter Tuning**:
   - Adjust `hidden_size` (64, 128, 256)
   - Try different `num_layers` (1, 2, 3)
   - Experiment with `dropout` rates (0.2-0.5)

2. **Training Settings**:
   - Increase `batch_size` if GPU memory allows
   - Adjust `learning_rate` (try 0.0001 or 0.01)
   - Increase `num_epochs` for longer training

3. **Data Preprocessing**:
   - Feature engineering: create velocity/acceleration from joint positions
   - Handle missing values (if any) using `utils/preprocessing.py`
   - Data augmentation with noise (see `utils/preprocessing.py`)

4. **Model Ensemble**:
   - Train multiple models with different seeds
   - Average predictions for better results

## 🔧 Troubleshooting

**CUDA out of memory**: Reduce `batch_size` in config

**Poor performance**: 
- Increase model capacity (`hidden_size`, `num_layers`)
- Train for more epochs
- Reduce regularization (`dropout`, `weight_decay`)

**Overfitting**:
- Increase `dropout`
- Add more weight decay
- Use data augmentation
- Reduce model complexity

**Data loading errors**: 
- Verify CSV paths in config
- Check data format matches expected structure
- Ensure `sample_index` and `time` columns exist

## 🛠️ Advanced Usage

### Resume Training
```bash
python train.py --config configs/gru_config.yaml --resume checkpoints/gru_best.pth
```

### Custom Preprocessing
Modify `utils/preprocessing.py` to add:
- Missing value handling
- Data augmentation
- Feature engineering

### Export Predictions with Probabilities
Modify `predict.py` to save probability scores alongside class predictions.

## 📝 File Formats

### Input CSV Format (train/test):
```
sample_index,time,pain_survey_1,pain_survey_2,...,joint_00,joint_01,...
0,0,0.5,0.3,...,45.2,32.1,...
0,1,0.5,0.3,...,45.5,32.3,...
...
0,179,0.6,0.4,...,46.1,33.0,...
1,0,0.2,0.1,...,38.4,29.7,...
```

### Labels CSV Format:
```
sample_index,pain_level
0,low_pain
1,no_pain
2,high_pain
...
```

### Submission CSV Format:
```
sample_index,pain_level
0,no_pain
1,low_pain
...
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project structure is inspired by best practices in deep learning competitions and research projects.

---

**Good luck with the challenge! ⚓**
