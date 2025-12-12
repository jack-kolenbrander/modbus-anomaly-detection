"""
Series of functions to initialize dataset and select valuable/important features.
"""

import pandas as pd
import numpy as np
import seaborn as sns

def initalize_modbus_dataset(dataset):
    """
    Function to drop nonvaluable features in the modbus dataset.
    """

    # Create copy of initial dataset
    df = dataset.copy()

    EXCLUSIONS = [
        'IP_SRC', 'IP_DST',
        'Ether_src', 'Ether_dst',

        
    ]

    

def select_features(dataset):
    """
    Function to select features for analysis.
    """


