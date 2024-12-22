import pandas as pd
import numpy as np 
from abc import ABC
from typing import List, Any
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class Transforms(ABC):
    
    @staticmethod
    def select_columns(df: pd.DataFrame, features: str | List[str] ):
        return df[features]
    
    @staticmethod
    def all_cat_column_label_encode(df: pd.DataFrame, num_unique_values: int = 5):
        copy_dset = df.copy()
        for column in copy_dset.columns:
            unique = copy_dset[column].unique()
            
            if len(unique) <= num_unique_values:
                encoded = range(0, len(unique))
                copy_dset[column] = copy_dset[column].replace(unique, encoded)
            
        return copy_dset
    
    @staticmethod
    def fill_na(df: pd.DataFrame, column: str | None = None, value: Any = None):
        
        if column and value:
            df.fillna({column: value}, inplace=True)
            return df
        else:
            for col in df.columns:
                value = np.random.choice(df[col])
                df[col] = df[col].fillna(value=value)
            
            return df
     
    

