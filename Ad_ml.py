import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_and_preprocess_data():
    # Load data
    data_dir = "/Users/aj/Documents/Projects/Anomaly Detection/dataset/"
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Load and combine all parquet files
    dfs = []
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
            print(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    print("\nInitial dataframe shape:", full_df.shape)

    # --- Reduce dataset size to 50% to prevent performance issues ---
    print(f"Original dataset has {len(full_df)} rows. Reducing to 50%...")
    full_df = full_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    print(f"Reduced dataset shape: {full_df.shape}")
    
    # Handle missing values
    print("\nMissing values before cleaning:")
    print(full_df.isnull().sum().sort_values(ascending=False).head(10))
    
    # Drop columns with all missing values
    full_df = full_df.dropna(axis=1, how='all')
    
    # Drop timestamp columns for simplicity (can be engineered later if needed)
    if 'Timestamp' in full_df.columns:
        full_df = full_df.drop('Timestamp', axis=1)
    
    # Drop other high-cardinality or unnecessary columns
    cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port']
    full_df = full_df.drop(columns=[col for col in cols_to_drop if col in full_df.columns])
    
    # Encode categorical variables if any exist
    categorical_cols = full_df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print("\nEncoding categorical columns:", list(categorical_cols))
        le = LabelEncoder()
        for col in categorical_cols:
            full_df[col] = le.fit_transform(full_df[col].astype(str))
    
    # Replace infinite values with NaN so they can be filled
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill remaining missing values with column mean
    full_df = full_df.fillna(full_df.mean())
    
    print("\nFinal dataframe shape after preprocessing:", full_df.shape)
    return full_df

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    This is a supervised model, which is better since we have labels.
    """
    print("\nTraining RandomForestClassifier model...")
    # Using a supervised classifier because labels are available.
    # class_weight='balanced' helps to handle the imbalanced dataset.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics
    """
    print("\nModel evaluation:")
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Convert multi-class labels to binary (0 for normal, 1 for any anomaly)
    # This assumes that class '0' is the normal class.
    y_test_binary = (y_test != 0).astype(int)
    y_pred_binary = (y_pred != 0).astype(int)
    print("\n(Note: Labels are converted to binary: 0=Normal, 1=Anomaly for evaluation)")

    # Calculate and print metrics
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary, target_names=['Normal', 'Anomaly']))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Separate features and target (if exists)
    target_col = 'Label' if 'Label' in df.columns else None
    X = df.drop(columns=[target_col]) if target_col else df
    y = df[target_col] if target_col else None

    if y is None:
        print("No 'Label' column found. Cannot perform supervised learning.")
        return

    # --- Create a Train/Test Split (80% train, 20% test) ---
    # This is crucial to evaluate how the model performs on unseen data.
    print("\nSplitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Standardize features
    # Fit the scaler on the training data ONLY, then transform both train and test data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = train_model(X_train, y_train)
    
    # Evaluate the model on the unseen test set
    print("\n--- Evaluating on the TEST set ---")
    evaluate_model(model, X_test, y_test)
    
    # Feature importance (for interpretation)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(feature_importance.head(10))

if __name__ == "__main__":
    main()
