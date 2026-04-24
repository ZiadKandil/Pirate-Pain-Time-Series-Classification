# Pirate Pain Time-Series Classification

This repository contains a full PyTorch pipeline for a time-series classification task. The main goal is to classify sequences of data into pain levels: `no_pain`, `low_pain`, or `high_pain`.

## Project Structure

- `data/`: Contains the training and testing datasets (CSV format).
- `configs/`: YAML configuration files containing hyperparameters for model architecture and the training process.
- `models/`: PyTorch definitions for sequential models, specifically GRU and LSTM variants.
- `utils/`: Contains preprocessing logic and dataset loading (`dataset.py`), alongside metric calculation (`metrics.py`).
- `explore_data.py`: Exploratory Data Analysis (EDA) script to analyze sequence formats, check for missing values, understand data distributions, and calculate average sequence lengths.
- `train.py`: The main script to train the model, save checkpoints, and output validation metrics (F1 Macro and Accuracy).
- `predict.py`: An inference script to generate categorical predictions on test data and save them in a submission-ready CSV format.

## Data Preprocessing (`utils/dataset.py`)

Several essential preprocessing steps were implemented to clean and unify the data formats before feeding it to the PyTorch models:

1. **Categorical Features Conversion**: Feature variables explicitly stored as strings (e.g., `n_legs`, `n_hands`, `n_eyes` containing `"one"`, `"two"`) are converted into numerical representations dynamically using Pandas categories (`.astype('category').cat.codes`) to ensure tensor operations don't throw `ValueError` strings.
2. **Missing Values Strategies**: Configurations allow applying variable strategies (mean replacement, forward fill, backward fill) to effectively handle NaN gaps in sequences.
3. **Data Scaling**: `StandardScaler` from scikit-learn is fit entirely on the training split to avoid data leakage, and simply applied (`.transform()`) to validation/test sets to bound numerical scales efficiently.
4. **Data Augmentation**: Time-series specific augmentations like Gaussian Noise injection can be controlled through the configuration files to increase data distribution density if needed.

## Models Strategy

- Time-series sequence padding and truncation are hard-set up to `seq_length=160`.
- Implemented GRU (`gru_model.py`) and LSTM (`lstm_model.py`) architectures as primary baselines. The flexibility of architectures is handled smoothly via parsing `yaml` configurations.
- Focus heavily on optimizing towards the **F1 Macro** evaluation metric due to heavily inherent class imbalance mapping observed early on (`no_pain` mostly dominating `low_pain` and `high_pain`).

## How to Run

### 1. Training

To train the target model based on a configuration file, run:

```bash
python train.py --config configs/gru_config.yaml
```

To **resume training** or start from a previously saved checkpoint (useful if training was interrupted), simply pass the `--resume` argument:

```bash
python train.py --config configs/gru_config.yaml --resume checkpoints/gru_best.pth
```

During training, the `train.py` loop will:
- Partition the dataset appropriately to prevent validation leakage.
- Update model weights and check against Early Stopping criteria.
- Periodically dump the best weights as `checkpoints/<model>_best.pth`.

### 2. Prediction / Inference

To perform prediction using a saved checkpoint over hidden test sequences, execute the inference script:

```bash
python predict.py --config configs/gru_config.yaml --checkpoint checkpoints/gru_best.pth --output submission.csv
```

This transforms numerical predictions (`0`, `1`, `2`) back to human-readable target labels (`high_pain`, `low_pain`, `no_pain`) using the final mapped dictionary before saving it directly to CSV.
