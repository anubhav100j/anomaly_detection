from datasets import load_dataset
import pandas as pd
import glob
import os

# 1. Set the path to your local dataset directory.
# `os.path.expanduser` correctly resolves "~" to your home directory.
data_dir = "/Users/aj/Documents/Projects/Anomaly Detection/dataset/"

# 2. Find all data files in that directory.
#data_file = "/Users/aj/Documents/Projects/Anomaly Detection/dataset/logs-parquet_part2.parquet"

all_files = glob.glob(os.path.join(data_dir, "*.parquet"))

if not all_files:
    print(f"No parquet files found in {data_dir}")
else:
    dfs = []
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
            print(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    print("Combined dataframe shape:", full_df.shape)
    print(full_df.head())