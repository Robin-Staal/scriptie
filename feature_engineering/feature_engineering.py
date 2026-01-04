import pandas as pd
import numpy as np

class FeatureEngineeringPipeline:
    def __init__(self, windows=(7, 14, 21, 28)):
        self.windows = windows
        
    def engineer_strength(self, strength_df: pd.DataFrame) -> pd.DataFrame:
        '''
        TODO: We need some scientifically valid way to quantify strength training load.
        
        For now, we use a naive approach: weight * repetitions.
        We construct strength_load per exercise and then aggregate it per day, per bodypart.
        From (exercise-level):
        - Weight
        - Repetitions
        - Bodypart
        
        To (day-level, per bodypart):
        - num_sets
        - total_strength_load
        - max_strength_load
        - avg_strength_load
        - max_weight
        - avg_weight
        - max_repetitions
        - avg_repetitions
        '''
        strength_df = strength_df.copy()
        strength_df['strength_load'] = strength_df['weight'] * strength_df['repetitions']
        strength_df_daily = strength_df.groupby(['Date', 'bodypart']).agg(
            num_sets=('strength_load', 'count'),
            total_strength_load=('strength_load', 'sum'),
            max_strength_load=('strength_load', 'max'),
            avg_strength_load=('strength_load', 'mean'),
            max_weight=('weight', 'max'),
            avg_weight=('weight', 'mean'),
            max_repetitions=('repetitions', 'max'),
            avg_repetitions=('repetitions', 'mean')
        ).reset_index()
        
        return strength_df_daily
        
    def engineer_endurance(self, endurance_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Training load of endurance sessions calculated as RPE * duration (in minutes).
        Which as proven to be valid and reliable by (Foster et al. (2001)) and many more.
        
        We construct training_load per session and then aggregate it per day.
        From (session-level):
        - RPE
        - Duration (in minutes)
        
        To (day-level):
        - num_sessions
        - total_load
        - max_load
        - avg_load
        - max_RPE
        - avg_RPE
        - max_duration
        - avg_duration
        '''
        endurance_df = endurance_df.copy()
        endurance_df['training_load'] = endurance_df['RPE'] * endurance_df['duration']
        endurance_df_daily = endurance_df.groupby('Date').agg(
            num_sessions=('training_load', 'count'),
            total_load=('training_load', 'sum'),
            max_load=('training_load', 'max'),
            avg_load=('training_load', 'mean'),
            max_RPE=('RPE', 'max'),
            avg_RPE=('RPE', 'mean'),
            max_duration=('duration', 'max'),
            avg_duration=('duration', 'mean')
        ).reset_index()
        
        return endurance_df_daily
    
    def transform(self, endurance_df: pd.DataFrame, strength_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Merges and engineers features from endurance, strength, and wellness dataframes.
        Endurance dataframe must contain 'Date', 'RPE', 'duration'.
        Strength dataframe must contain 'Date', 'weight', 'repetitions', 'bodypart'.
        Additional dataframe must contain 'Date' and other attributes (e.g., wellness scores, steps, food) that need to be considered.
        '''
        df = df.copy()

        for w in self.windows:
            df[f"load_{w}d"] = df["load_1d"].rolling(w).mean()
            df[f"sessions_{w}d"] = df["sessions_1d"].rolling(w).sum()

        # ACWR
        df["acwr_7_28"] = df["load_7d"] / df["load_28d"]

        # Lagged wellness (important!)
        df["target_lag1"] = df["target"].shift(1)

        return df.dropna().reset_index(drop=True)
