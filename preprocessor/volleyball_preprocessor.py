import pandas as pd
from preprocessor.base_preprocessor import Preprocessor

class VolleyBallPreprocessor(Preprocessor):
    def __init__(self, exercise_path, rpe_path, wellness_path, jumps_path, strength_path):
        self.exercise_path = exercise_path
        self.rpe_path = rpe_path
        self.wellness_path = wellness_path
        self.jumps_path = jumps_path
        self.strength_data = strength_path
        
    def load(self):
        exercise_data = pd.read_csv(self.exercise_path, sep=';')
        strength_data = pd.read_csv(self.strength_data, sep=';')
        rpe_data = pd.read_csv(self.rpe_path, sep=';')
        wellness_data = pd.read_csv(self.wellness_path, sep=';')
        jumps_data = pd.read_csv(self.jumps_path, sep=';')
        
        # Exercise-type is ignored for now, we only look at RPE and duration
        training_dates = (
            exercise_data[['TrainingID', 'Date']]
            .drop_duplicates(subset='TrainingID')
        )
        rpe_data = rpe_data.drop(columns=['Date'])
        
        rpe_data = pd.merge(
            rpe_data,
            training_dates,
            on='TrainingID',
            how='left'
        )
        rpe_data = rpe_data[['Date', 'RPE', 'Duration']]
        
        # Make date notation consistent
        rpe_data['Date'] = pd.to_datetime(rpe_data['Date'])
        wellness_data['Date'] = pd.to_datetime(wellness_data['Date'])
        strength_data['Date'] = pd.to_datetime(strength_data['Date'])
        