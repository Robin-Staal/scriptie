import pandas as pd
import numpy as np

class FeatureEngineeringPipeline:
    def __init__(self, windows=(3, 5, 7, 14, 21, 28), lags=(1, 2, 3)):
        self.windows = windows
        self.lags = lags
        self.aggregates = {
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'std': np.std,
        }
        
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
        strength_df['strength_load'] = strength_df['Weight'] * strength_df['Reps']
        strength_df_daily = strength_df.groupby(['Date', 'Exercise']).agg(
            num_sets=('strength_load', 'count'),
            total_strength_load=('strength_load', 'sum'),
            max_strength_load=('strength_load', 'max'),
            avg_strength_load=('strength_load', 'mean'),
            max_weight=('Weight', 'max'),
            avg_weight=('Weight', 'mean'),
            max_repetitions=('Reps', 'max'),
            avg_repetitions=('Reps', 'mean')
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
        endurance_df['training_load'] = endurance_df['RPE'] * endurance_df['Duration']
        endurance_df_daily = endurance_df.groupby('Date').agg(
            num_sessions=('training_load', 'count'),
            total_load=('training_load', 'sum'),
            max_load=('training_load', 'max'),
            avg_load=('training_load', 'mean'),
            max_RPE=('RPE', 'max'),
            avg_RPE=('RPE', 'mean'),
            max_duration=('Duration', 'max'),
            avg_duration=('Duration', 'mean')
        ).reset_index()
        
        return endurance_df_daily
    
    def construct_aggregates(self, daily_series_df: pd.DataFrame) -> pd.DataFrame:
        '''
        For each (temporal window, aggregate function) pair, constructs aggregate features.
        E.g., for a window of 7 days and 'mean' aggregate function, constructs a feature that contains the rolling mean over the past 7 days.
        '''
        df = daily_series_df.copy()
        df = df.set_index('Date').sort_index()

        new_features = {}

        for col in df.columns:
            if col == 'wellness_total':
                continue
            series = df[col]
            for w in self.windows:
                # Shift by 1 to exclude current day (t) and start window at t-1
                rolling = series.shift(1).rolling(window=w)
                for agg_name, agg_func in self.aggregates.items():
                    feature_name = f"{col}_{agg_name}_{w}d"
                    new_features[feature_name] = rolling.apply(
                        agg_func, raw=True
                    )

        features_df = pd.DataFrame(new_features, index=df.index)

        # concatonate the new features
        df = pd.concat([df, features_df], axis=1)

        return df.reset_index()

    
    def transform(self, endurance_df: pd.DataFrame, strength_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Endurance dataframe must contain 'Date', 'RPE', 'duration'.
        Strength dataframe must contain 'Date', 'weight', 'repetitions', 'bodypart'.
        Additional dataframe must contain 'Date' and other attributes (e.g., wellness scores, steps, food) that need to be considered.
        
        1) Converts endurance and strength data to daily aggregates.
        2) Merges endurance, strength, and additional data on 'Date'.
        3) Constructs aggregate features for each (temporal window, aggregate function) pair.
        TODO: 4) Applies time-shifting on each feature to create lagged features.
        5) Creates additional domain-specific features like ACWR.
        '''
        endurance_daily = self.engineer_endurance(endurance_df)
        strength_daily = self.engineer_strength(strength_df)
        daily_series_df = pd.merge(endurance_daily, strength_daily, on='Date', how='outer')
        daily_series_df = pd.merge(daily_series_df, additional_df, on='Date', how='outer')
        daily_series_df = daily_series_df.copy()
        
        daily_series_df.drop(columns=['Exercise'], inplace=True, errors='ignore')
        daily_series_engineered_df = self.construct_aggregates(daily_series_df)
        
        # Apply ACWR calculation
        if 'total_load' in daily_series_engineered_df.columns:
            daily_series_engineered_df = daily_series_engineered_df.sort_values('Date')
            # ACWR = acute load (7-day rolling mean) / chronic load (7d-28d rolling mean)
            daily_series_engineered_df['ACWR'] = (
                daily_series_engineered_df['total_load_mean_7d'] /
                (daily_series_engineered_df['total_load_mean_28d'])
            )
            print("ACWR feature created:", daily_series_engineered_df['ACWR'].head(50))
            
        # write the dataframe to a csv for inspection
        daily_series_engineered_df.to_csv("daily_series_engineered.csv", index=False)
        
        return daily_series_engineered_df
