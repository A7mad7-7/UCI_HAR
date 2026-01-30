import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Download & Extract Dataset (Auto-Setup)
# ==========================================
url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
zip_filename = "UCI HAR Dataset.zip"
dataset_dir = "UCI HAR Dataset"

if not os.path.exists(dataset_dir):
    if os.path.exists(zip_filename):
        print(f"Found local dataset zip: {zip_filename}. Extracting... ⏳")
        with zipfile.ZipFile(zip_filename, 'r') as z:
            z.extractall(".")
        print("Extraction Completed! ✅")
    else:
        print("Downloading UCI HAR Dataset... ⏳")
        r = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(".")
        print("Download & Extraction Completed! ✅")

# ==========================================
# 2. Load Features & Labels
# ==========================================
print("Loading Data... 📂")

# Load feature names (561 features)
features_path = os.path.join(dataset_dir, 'features.txt')
features_df = pd.read_csv(features_path, sep=r'\s+', header=None, names=['index', 'feature_name'])
feature_names = features_df['feature_name'].values

# Load Training Data
X_train_path = os.path.join(dataset_dir, 'train', 'X_train.txt')
y_train_path = os.path.join(dataset_dir, 'train', 'y_train.txt')
X_train_orig = pd.read_csv(X_train_path, sep=r'\s+', header=None).values
y_train_orig = pd.read_csv(y_train_path, header=None).values.flatten()

# Load Test Data
X_test_path = os.path.join(dataset_dir, 'test', 'X_test.txt')
y_test_path = os.path.join(dataset_dir, 'test', 'y_test.txt')
X_test_orig = pd.read_csv(X_test_path, sep=r'\s+', header=None).values
y_test_orig = pd.read_csv(y_test_path, header=None).values.flatten()

# ==========================================
# 3. Combine & Re-split Data (65% train, 20% cv, 15% test)
# ==========================================
print("Splitting Data (65% train, 20% cv, 15% test)... 📊")

# Combine all data
X_all = np.vstack([X_train_orig, X_test_orig])
y_all = np.concatenate([y_train_orig, y_test_orig])

# Convert labels from 1-6 to 0-5 for model compatibility
y_all = y_all - 1

# First split: 85% (train+cv) vs 15% (test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)

# Second split: 65% train, 20% cv from the 85%
# 20/85 ≈ 0.235 of the remaining data for cv
X_train, X_cv, y_train, y_cv = train_test_split(
    X_temp, y_temp, test_size=0.235, random_state=42, stratify=y_temp
)

# ==========================================
# 4. Save Processed Data
# ==========================================
data_dir = "processed_data"
os.makedirs(data_dir, exist_ok=True)

np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'X_cv.npy'), X_cv)
np.save(os.path.join(data_dir, 'y_cv.npy'), y_cv)
np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

print(f"\n✅ Data saved to '{data_dir}/' folder!")

# ==========================================
# 5. Data Summary
# ==========================================
print("\n" + "=" * 50)
print("📊 DATA SUMMARY")
print("=" * 50)
print(f"🔹 Total samples: {len(X_all)}")
print(f"🔹 Number of features: {X_all.shape[1]}")
print(f"🔹 Number of classes: {len(np.unique(y_all))}")
print("-" * 50)
print(f"🔹 Training set:     {len(X_train)} samples ({len(X_train)/len(X_all)*100:.1f}%)")
print(f"🔹 Cross-val set:    {len(X_cv)} samples ({len(X_cv)/len(X_all)*100:.1f}%)")
print(f"🔹 Test set:         {len(X_test)} samples ({len(X_test)/len(X_all)*100:.1f}%)")
print("=" * 50)