import pandas as pd
import glob
import os

# Path to your local dataset directory
data_dir = os.path.expanduser("")


print("Loading dataset...")
try:
    ds = load_dataset("")
    print("Dataset loaded successfully!")
    print(f"Dataset splits: {list(ds.keys())}")
    
    # Print first example to verify
    print("\nFirst example:")
    print(ds['train'][0])
    
    # Access the train split
    train_split = ds["train"]
    print("\nFirst example from the dataset")
    print(train_split[0])
    
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("\nPlease check:")
    print("1. Your internet connection")
    print("2. If you need to authenticate with Hugging Face")
    print("3. If the dataset exists and is accessible")
    print("4. If the dataset schema matches your requirements")
    
    # Try to clean up the cache if it exists
    try:
        cache_dir = os.environ["HF_DATASETS_CACHE"]
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("\nCache cleaned up. Try running the script again.")
    except Exception as cleanup_error:
        print(f"Error cleaning up cache: {str(cleanup_error)}")
        