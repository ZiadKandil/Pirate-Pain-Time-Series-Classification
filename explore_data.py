import pandas as pd
import argparse
import os

def explore_data(data_path, labels_path=None):

    #load data
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return
    df = pd.read_csv(data_path)

    print('1- Basic Dataset Information:')
    df.info()
    print('\n')

    print('2- First 5 rows:')
    print(df.head())
    print('\n')

    print('3- Statistics')
    print(df.describe())
    print('\n')

    print('4- Missing values')
    missing_values = df.isna().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else 'No missing values')
    print('\n')

    print('5- Unique entries per feature')
    print(df.nunique())
    print('\n')

    print('6- Sequence lengths info')
    seq_lengths = df.groupby('sample_index').size()
    print(f'Number of samples: {len(seq_lengths)}')
    print(f'Average sequence length: {seq_lengths.mean()}')
    print(f'Min sequence length: {seq_lengths.min()}')
    print(f'Max sequence length: {seq_lengths.max()}')
    print('\n')

    # Explore class inbalance if labels are provided
    if labels_path and os.path.exists(labels_path):

        labels_df = pd.read_csv(labels_path)
        print('Class distribution:')
        class_counts = labels_df['label'].value_counts()
        class_percentages = labels_df['label'].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Count': class_counts, 'Percentage': class_percentages})
        print(dist_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploring Data")
    parser.add_argument('--data', type=str, default='data/train_data.csv', help='Path to the data CSV file')
    parser.add_argument('--labels', type=str, default='data/train_labels.csv', help='Path to the labels CSV file')
    
    args = parser.parse_args()
    explore_data(args.data, args.labels)
