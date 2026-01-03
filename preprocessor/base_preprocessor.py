from abc import ABC, abstractmethod
import pandas as pd

class Preprocessor(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Returns a DataFrame with daily exercise data.
        Optionally returns questionnaire data if available.
        """
        pass