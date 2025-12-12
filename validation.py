import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- CONFIG ----------

TRAIN_RAW = "train.csv"
TEST_RAW = "test.csv"
TRAIN_PROC = "train_processed.csv"
TEST_PROC = "test_processed.csv"
CV_SPLITS = "cv_splits.pkl"

PLOT_FILE = "validation_plots.png"


# ---------- HELPER PLOTS ----------

def plot_preprocessing(train_raw, train_proc, out_path=PLOT_FILE):
    """Create a single figure summarizing NDVI, height, and date transforms."""
    print("📊 Creating validation plots...")

    # sample to keep plots readable
    sample_raw = train_raw.sample(min(2000, len(train_raw)), random_state=42)
    sample_proc = train_proc.loc[sample_raw.index]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Preprocessing Validation", fontsize=16, fontweight="bold")

    # 1. NDVI normalization
    ax = axes[0, 0]
    ax.scatter(sample_raw["Pre_GSHH_NDVI"], sample_proc["ndvi_norm"], alpha=0.5)
    ax.set_title("NDVI normalization")
    ax.set_xlabel("Pre_GSHH_NDVI (raw)")
    ax.set_ylabel("ndvi_norm [0,1]")
    ax.grid(True, alpha=0.3)

    # 2. Height log + std
    ax = axes[0, 1]
    ax.scatter(sample_raw["Height_Ave_cm"], sample_proc["height_norm"], alpha=0.5)
    ax.set_title("Height log + standardization")
    ax.set_xlabel("Height_Ave_cm (raw)")
    ax.set_ylabel("height_norm")
    ax.grid(True, alpha=0.3)

    # 3. Date cyclical encoding
    ax = axes[0, 2]
    doy = pd.to_datetime(sample_raw["Sampling_Date"]).dt.dayofyear
    ax.scatter(doy, sample_proc["date_sin"], alpha=0.5, label="sin")
    ax.scatter(doy, sample_proc["date_cos"], alpha=0.5, label="cos")
    ax.set_title("Date cyclical encoding")
    ax.set_xlabel("day of year")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Height raw distribution
    ax = axes[1, 0]
    sample_raw["Height_Ave_cm"].hist(bins=30, alpha=0.8, ax=ax)
    ax.set_title("Height raw distribution")
    ax.set_xlabel("Height_Ave_cm")
    ax.set_ylabel("count")

    # 5. Height normalized distribution
    ax = axes[1, 1]
    sample_proc["height_norm"].hist(bins=30, alpha=0.8, ax=ax)
    ax.set_title("Height normalized distribution")
    ax.set_xlabel("height_norm")
    ax.set_ylabel("count")

    # 6. leave last panel for text summary
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plots to {out_path}")


# ---------- CHECKS ----------

def check_train():
    print("=== TRAIN CHECK ===")
    if not (os.path.exists(TRAIN_RAW) and os.path.exists(TRAIN_PROC)):
        print("❌ train.csv or train_processed.csv missing")
        return

    raw = pd.read_csv(TRAIN_RAW)
    proc = pd.read_csv(TRAIN_PROC)

    print("raw shape:", raw.shape)
    print("proc shape:", proc.shape)

    # same number of rows
    assert len(raw) == len(proc), "row count mismatch between raw and processed train"

    # key columns preserved
    key_cols = [
        "sample_id", "image_path", "Sampling_Date",
        "State", "Species", "target_name", "target"
    ]
    print("\nKey columns (first 3 rows, raw vs processed):")
    print("RAW:")
    print(raw.loc[:2, key_cols])
    print("PROC:")
    print(proc.loc[:2, key_cols])

    # engineered columns
    eng_cols = [
        "ndvi_norm", "height_norm",
        "date_sin", "date_cos", "date_sin2", "date_cos2",
        "state_id", "species_id",
        "ndvi_height_interaction", "seasonal_ndvi",
    ]
    missing = [c for c in eng_cols if c not in proc.columns]
    print("\nMissing engineered columns:", missing)
    assert not missing, "some engineered columns are missing in train_processed.csv"

    print("\nNaNs in engineered columns:")
    print(proc[eng_cols].isna().sum())

    print("\nRanges / stats:")
    print("ndvi_norm range:", float(proc["ndvi_norm"].min()),
          float(proc["ndvi_norm"].max()))
    print("height_norm mean/std:",
          float(proc["height_norm"].mean()),
          float(proc["height_norm"].std()))

    # per‑image: 5 rows per physical image ID
    base_id = raw["sample_id"].str.split("__").str[0]
    counts = base_id.value_counts()
    print("\nrows per physical image (base sample_id):")
    print(counts.value_counts())  # expect 5 as dominant count

    # example image
    example_id = base_id.iloc[0]
    print("\nExample physical image rows:")
    print(proc[base_id == example_id][
        ["sample_id", "image_path", "target_name", "target"]
    ])

    # image paths exist
    paths = proc["image_path"].unique()
    missing_files = [p for p in paths if not os.path.exists(p)]
    print("\nUnique image paths:", len(paths))
    print("Missing image files:", len(missing_files))
    if missing_files:
        print("First few missing:", missing_files[:5])

    # make plots
    plot_preprocessing(raw, proc, PLOT_FILE)


def check_test():
    print("\n=== TEST CHECK ===")
    if not (os.path.exists(TEST_RAW) and os.path.exists(TEST_PROC)):
        print("❌ test.csv or test_processed.csv missing")
        return

    raw = pd.read_csv(TEST_RAW)
    proc = pd.read_csv(TEST_PROC)

    print("raw shape:", raw.shape)
    print("proc shape:", proc.shape)

    # core columns preserved
    print("\nHead (raw):")
    print(raw.head())
    print("\nHead (processed core cols):")
    print(proc[["sample_id", "image_path", "target_name"]].head())

    eng_cols = [
        "ndvi_norm", "height_norm",
        "date_sin", "date_cos", "date_sin2", "date_cos2",
        "state_id", "species_id",
        "ndvi_height_interaction", "seasonal_ndvi",
    ]
    missing = [c for c in eng_cols if c not in proc.columns]
    print("\nMissing engineered/dummy columns in test_processed:", missing)
    assert not missing, "test_processed.csv missing engineered dummy columns"

    print("\nNaNs in test_processed:")
    print(proc.isna().sum())

    print("\nUnique values for dummy engineered columns (should be constants):")
    for c in eng_cols:
        vals = np.unique(proc[c].values)
        print(c, vals[:5])


def check_cv():
    print("\n=== CV SPLITS CHECK ===")
    if not (os.path.exists(CV_SPLITS) and os.path.exists(TRAIN_PROC)):
        print("❌ cv_splits.pkl or train_processed.csv missing")
        return

    proc = pd.read_csv(TRAIN_PROC)
    with open(CV_SPLITS, "rb") as f:
        splits = pickle.load(f)

    print("num folds:", len(splits))
    n = len(proc)
    for s in splits:
        tr = s["train_idx"]
        va = s["val_idx"]
        fold = s["fold"]
        print(f"fold {fold}: train={len(tr)}, val={len(va)}")
        assert max(tr) < n and max(va) < n, "index out of range in CV splits"
        assert len(set(tr).intersection(set(va))) == 0, "train/val overlap in CV splits"


def main():
    check_train()
    check_test()
    check_cv()
    print("\n✅ validation_processing.py finished successfully")


if __name__ == "__main__":
    main()
