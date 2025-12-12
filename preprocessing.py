import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class CSIROPreprocessor:
    """
    TRAIN rows:
      sample_id, image_path, Sampling_Date, State, Species,
      Pre_GSHH_NDVI, Height_Ave_cm, target_name, target

    TEST rows:
      sample_id, image_path, target_name
    """

    def __init__(self):
        self.ndvi_min = 0.28
        self.ndvi_max = 0.89
        self.height_scaler = StandardScaler()
        self.state_map = {}
        self.species_map = {}
        self.num_states = 0
        self.num_species = 0
        self.cv_splits = None

    def fit(self, df_train: pd.DataFrame):
        print("FITTING ON TRAIN...")
        print("  train shape:", df_train.shape)

        # Height log + std
        h_raw = df_train["Height_Ave_cm"].values
        h_log = np.log(h_raw + 0.1)
        self.height_scaler.fit(h_log.reshape(-1, 1))
        print(f"  height_log mean={self.height_scaler.mean_[0]:.4f}, "
              f"std={self.height_scaler.scale_[0]:.4f}")

        # Categorical maps
        states = sorted(df_train["State"].dropna().unique())
        species = sorted(df_train["Species"].dropna().unique())
        self.state_map = {s: i for i, s in enumerate(states)}
        self.species_map = {s: i for i, s in enumerate(species)}
        self.num_states = len(states)
        self.num_species = len(species)
        print("  states:", states)
        print("  species:", species)

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        print("TRANSFORMING TRAIN:", len(df), "rows")
        out = df.copy()

        # NDVI → [0,1]
        out["ndvi_norm"] = np.clip(
            (df["Pre_GSHH_NDVI"] - self.ndvi_min) / (self.ndvi_max - self.ndvi_min),
            0.0, 1.0
        )

        # Height log + std
        h_log = np.log(df["Height_Ave_cm"].fillna(1.0) + 0.1)
        out["height_norm"] = self.height_scaler.transform(
            h_log.to_numpy().reshape(-1, 1)
        ).ravel()

        # Date cyclical
        dates = pd.to_datetime(df["Sampling_Date"])
        doy = dates.dt.dayofyear.to_numpy()
        out["date_sin"] = np.sin(2 * np.pi * doy / 365.0)
        out["date_cos"] = np.cos(2 * np.pi * doy / 365.0)
        out["date_sin2"] = np.sin(4 * np.pi * doy / 365.0)
        out["date_cos2"] = np.cos(4 * np.pi * doy / 365.0)

        # IDs
        out["state_id"] = df["State"].map(self.state_map).astype(int)
        out["species_id"] = df["Species"].map(self.species_map).astype(int)

        # Interactions
        out["ndvi_height_interaction"] = out["ndvi_norm"] * out["height_norm"]
        out["seasonal_ndvi"] = out["ndvi_norm"] * (out["date_sin"] + 1.0) / 2.0

        return out

    def transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TEST has only sample_id, image_path, target_name.
        We add dummy tabular features so the Dataset interface is the same.
        """
        print("TRANSFORMING TEST:", len(df), "rows")
        out = df.copy()
        out["ndvi_norm"] = 0.5
        out["height_norm"] = 0.0
        out["date_sin"] = 0.0
        out["date_cos"] = 1.0
        out["date_sin2"] = 0.0
        out["date_cos2"] = 1.0
        out["state_id"] = 0
        out["species_id"] = 0
        out["ndvi_height_interaction"] = 0.0
        out["seasonal_ndvi"] = 0.0
        return out

    def make_cv_splits(self, df_train: pd.DataFrame, n_splits: int = 5, seed: int = 42):
        print("MAKING CV SPLITS...")
        df = df_train.copy()
        doy = pd.to_datetime(df["Sampling_Date"]).dt.dayofyear
        season_bucket = pd.qcut(doy, q=4, labels=False, duplicates="drop")
        df["strata"] = df["State"].astype(str) + "_" + season_bucket.astype(str)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df["strata"])):
            splits.append({
                "fold": fold,
                "train_idx": tr_idx,
                "val_idx": va_idx,
                "train_size": len(tr_idx),
                "val_size": len(va_idx),
            })
            print(f"  fold {fold}: train={len(tr_idx)}, val={len(va_idx)}")
        self.cv_splits = splits

    def save(self, path: str):
        state = {
            "ndvi_min": self.ndvi_min,
            "ndvi_max": self.ndvi_max,
            "height_scaler": self.height_scaler,
            "state_map": self.state_map,
            "species_map": self.species_map,
            "num_states": self.num_states,
            "num_species": self.num_species,
            "cv_splits": self.cv_splits,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print("saved preprocessor to", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--train_out", default="train_processed.csv")
    parser.add_argument("--test_out", default="test_processed.csv")
    parser.add_argument("--prep_out", default="preprocessor.pkl")
    parser.add_argument("--cv_out", default="cv_splits.pkl")
    args = parser.parse_args()

    # load raw
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    prep = CSIROPreprocessor()
    prep.fit(train_df)

    train_proc = prep.transform_train(train_df)
    test_proc = prep.transform_test(test_df)
    prep.make_cv_splits(train_df)

    # save NEW files in current directory only
    train_proc.to_csv(args.train_out, index=False)
    test_proc.to_csv(args.test_out, index=False)
    print("saved:", args.train_out, "and", args.test_out)

    prep.save(args.prep_out)
    with open(args.cv_out, "wb") as f:
        pickle.dump(prep.cv_splits, f)
    print("saved:", args.prep_out, "and", args.cv_out)


if __name__ == "__main__":
    main()
