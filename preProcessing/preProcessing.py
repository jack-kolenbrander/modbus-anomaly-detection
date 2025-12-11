"""
Preprocessing script for Modbus network log datasets.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer

# Import dataset
def load_dataset(filepath):
    """
    Function to load datatset from CSV file
    """
    # Load Dataset
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    # Print dataset results
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
    return df 


def clean_dataset(dataset):
    """
    Function to clean dataset and ensure it is accurate and complete. 
    This function/process is necessary to ensure that there are no issues that could impact model performance. 
    
    Key steps include:
        1) Handling missing values
        2) Removing duplicate logs
        3) Standardizing formats of dates, strings, etc
    """

    # Make a copy of the original dataset
    df = dataset.copy()
    # Address missing values using imputation
    # Imputation is the process of filling in missing values, which can be achieved through using statistical processes such as mean, median, or mode.
    # Sklearn.impute library can be used to achieve this task
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    

    # Seperate X and y
    X = df.iloc[:, :-1] # Load everything but last column into X
    y_train = df[:, -1] # Load last column into Y



def encode_fields(dataset):
    """
    Function to encode necessary fields
    """
