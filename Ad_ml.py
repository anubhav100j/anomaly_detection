from datasets import load_dataset
import pandas as pd
import glob
import os

# Use a relative path to make the project portable
data_dir = "dataset/"

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
