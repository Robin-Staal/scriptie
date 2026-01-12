import pandas as pd
from preprocessor.base_preprocessor import Preprocessor

class VolleyBallPreprocessor(Preprocessor):
    def __init__(self, exercise_path, rpe_path, wellness_path, jumps_path, strength_path):
        self.exercise_path = exercise_path
        self.rpe_path = rpe_path
        self.wellness_path = wellness_path
        self.jumps_path = jumps_path
        self.strength_data = strength_path
        
    def preprocess_wellness(self, wellness_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Combine the four wellness scores into a single one.
        '''
        wellness_df = wellness_df.copy()
        wellness_df['wellness_total'] = (
            wellness_df['Difficultparticipating'] +
            wellness_df['Reducedtraining'] +
            wellness_df['Affectedperformance'] +
            wellness_df['Symptomscomplaints']
        )
        
        wellness_df.drop(columns=[
            'Difficultparticipating',
            'Reducedtraining',
            'Affectedperformance',
            'Symptomscomplaints',
            'Injury',
            'Sick/healthy',
            'Recovered'
        ], inplace=True)
        return wellness_df
        
    def preprocess_jumps(self, jumps_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Because the feature engineering class expects a time series with single entries per date, we aggregate the jumps data by date.
        
        PlayerID;Date;HeightInCm -> Date;AvgJumpHeightInCm;TotalJumps;MedianJumpHeightInCm
        '''
        
        jumps_df = jumps_df.copy()
        jumps_agg = jumps_df.groupby('Date').agg(
            AvgJumpHeight=('HeightInCm', 'mean'),
            TotalJumps=('HeightInCm', 'count')
        ).reset_index()
        jumps_agg['Date'] = pd.to_datetime(jumps_agg['Date'], dayfirst=True)
        return jumps_agg
    
    def preprocess_endurance(self, endurance_df: pd.DataFrame) -> pd.DataFrame:
        '''
        From (event/exercise-level):
        - Date
        - TrainingID
        - Duration (in minutes)
        - RPE
        
        To (training-level):
        - Date
        - Total duration (in minutes)
        - RPE
        '''
        endurance_df = endurance_df.copy()
        
        endurance_agg = endurance_df.groupby('TrainingID').agg(
            Date=('Date', 'first'),
            Duration=('Duration', 'sum'),
            RPE=('RPE', 'first'),
        ).reset_index()
        return endurance_agg
        
    def load(self):
        exercise_data = pd.read_csv(self.exercise_path, sep=';')
        strength_data = pd.read_csv(self.strength_data, sep=';')
        rpe_data = pd.read_csv(self.rpe_path, sep=';')
        wellness_data = pd.read_csv(self.wellness_path, sep=';')
        jumps_data = pd.read_csv(self.jumps_path, sep=';')
        jumps_data = self.preprocess_jumps(jumps_data)
        
        # Exercise-type is ignored for now, we only look at RPE and duration
        # We merge the RPE data onto the exercise data to get a per-training df
        exercise_data.drop(columns=['Duration'])
        endurance_data = pd.merge(
            exercise_data,
            rpe_data,
            on='TrainingID',
            how='left'
        )
        endurance_data['Date'] = pd.to_datetime(endurance_data['Date'], dayfirst=True)
        endurance_data['Duration'] = pd.to_timedelta(endurance_data['Duration_x'])
        endurance_data['Duration'] = endurance_data['Duration'].dt.total_seconds() / 60.0
        endurance_data = endurance_data[['Date', 'TrainingID', 'Duration', 'RPE']]
        endurance_data = self.preprocess_endurance(endurance_data)
        
        wellness_data = self.preprocess_wellness(wellness_data)
        wellness_data['Date'] = pd.to_datetime(wellness_data['Date'], dayfirst=True)
        wellness_data = pd.merge(
            wellness_data,
            jumps_data,
            on='Date',
            how='left'
        )
        
        # Make date notation consistent
        strength_data['Date'] = pd.to_datetime(strength_data['Date'], dayfirst=True)
        # the 'Weight' column has x,y data, convert to x.y
        strength_data['Weight'] = strength_data['Weight'].str.replace(',', '.').astype(float)
        
        return endurance_data, strength_data, wellness_data