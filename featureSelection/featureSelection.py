"""
Series of functions to initialize dataset and select valuable/important features.
"""

import pandas as pd
import numpy as np
import seaborn as sns

def initalizeModbusDataset(dataset):
    """
    Function to drop nonvaluable features in the modbus dataset.
    """

    # Create copy of initial dataset
    df = dataset.copy()

    # First, drop timestamp, ether_dst, and ether_src features
    # 


