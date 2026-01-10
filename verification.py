import pickle
import pandas as pd
import numpy as np

# Load the files
with open('cv_splits.pkl', 'rb') as f:
    cv_splits = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    prep_state = pickle.load(f)

# Load original train data
train_df = pd.read_csv('train.csv')
train_df['base_id'] = train_df['sample_id'].str.split('__').str[0]

print("=" * 60)
print("VERIFICATION REPORT")
print("=" * 60)

# Check 1: Total unique samples
unique_samples = train_df['base_id'].unique()
print(f"\n✓ Total unique samples: {len(unique_samples)}")
print(f"✓ Total rows: {len(train_df)}")

# Check 2: Verify each fold
print("\n" + "=" * 60)
print("FOLD-BY-FOLD CHECK")
print("=" * 60)

for fold_info in cv_splits:
    fold = fold_info['fold']
    tr_idx = fold_info['train_idx']
    va_idx = fold_info['val_idx']
    
    # Get base_ids for train and val
    train_base_ids = set(train_df.iloc[tr_idx]['base_id'].unique())
    val_base_ids = set(train_df.iloc[va_idx]['base_id'].unique())
    
    # Check for overlap
    overlap = train_base_ids & val_base_ids
    
    print(f"\nFold {fold}:")
    print(f"  Train: {len(tr_idx)} rows, {len(train_base_ids)} unique samples")
    print(f"  Val:   {len(va_idx)} rows, {len(val_base_ids)} unique samples")
    
    if len(overlap) > 0:
        print(f"  ❌ LEAKAGE DETECTED: {len(overlap)} overlapping samples!")
        print(f"     Overlapping IDs: {list(overlap)[:5]}...")
    else:
        print(f"  ✅ NO OVERLAP - Clean split!")

# Check 3: Verify all samples are used
print("\n" + "=" * 60)
print("COVERAGE CHECK")
print("=" * 60)

all_train_samples = set()
all_val_samples = set()

for fold_info in cv_splits:
    tr_idx = fold_info['train_idx']
    va_idx = fold_info['val_idx']
    
    all_train_samples.update(train_df.iloc[tr_idx]['base_id'].unique())
    all_val_samples.update(train_df.iloc[va_idx]['base_id'].unique())

print(f"Samples that appear in any train fold: {len(all_train_samples)}")
print(f"Samples that appear in any val fold: {len(all_val_samples)}")
print(f"Total unique samples: {len(unique_samples)}")

if len(all_val_samples) == len(unique_samples):
    print("✅ All samples used for validation at least once!")
else:
    print(f"⚠️ {len(unique_samples) - len(all_val_samples)} samples never validated!")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

# Final verdict
all_clean = all(
    len(set(train_df.iloc[fold_info['train_idx']]['base_id'].unique()) & 
        set(train_df.iloc[fold_info['val_idx']]['base_id'].unique())) == 0
    for fold_info in cv_splits
)

if all_clean:
    print("✅✅✅ ALL FOLDS CLEAN - NO DATA LEAKAGE DETECTED!")
    print("✅ Safe to upload and retrain!")
else:
    print("❌❌❌ LEAKAGE FOUND - DO NOT USE THESE FILES!")
