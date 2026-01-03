import pandas as pd
import numpy as np

class FeatureEngineeringPipeline:
    def __init__(self, windows=(7, 14, 21, 28)):
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for w in self.windows:
            df[f"load_{w}d"] = df["load_1d"].rolling(w).mean()
            df[f"sessions_{w}d"] = df["sessions_1d"].rolling(w).sum()

        # ACWR
        df["acwr_7_28"] = df["load_7d"] / df["load_28d"]

        # Lagged wellness (important!)
        df["target_lag1"] = df["target"].shift(1)

        return df.dropna().reset_index(drop=True)
