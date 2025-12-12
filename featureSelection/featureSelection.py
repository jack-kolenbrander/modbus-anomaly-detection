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
        # IP and MAC addresses
        'IP_src', #IP source address 
        'IP_dst', #IP Destination address
        'Ether_src', # Source ethernet address
        'Ether_dst', # Destination ethernet address
        # Protocol values
        'Ether_type', # Field that identifies upper-layer protcol or payload type
        'IP_version', # Field that identifies IP version
        'IP_proto', # Field that identifies protocol (TCP, UDP, ICMP, etc)
        'IP_chksum', # IP layer checksum
        'TCP_sport', # TCP source port
        'TCP_dport', # TCP Destination port
        'TCP_seq', # TCP sequence number
        'TCP_ack', # Expected TCP sequence umber
        'TCP_dataofs', # TCP offest 
        'TCP_reserved', # TCP reserved field
        'TCP_chksum', # TCP Checksum
        'TCP_urgptr', # Used when urgent bit has been set
        'TCP_options', # TCP optional field
    ]

    

def select_features(dataset):
    """
    Function to select features for analysis.
    """


