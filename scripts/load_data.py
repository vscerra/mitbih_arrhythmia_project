# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:41:44 2025

@author: vscerra
"""

import os
import wfdb
import numpy as np


def download_mitbih(destination: str = "data/raw", records: list = None):
    """
    Download selected records from the MIT-BIH Arrhythmia Database
    """
    if records is None:
        # A common subset used in research 
        records = ["100", "101", "102", "103", "104"]
        
    os.makedirs(destination, exist_ok = True)
    for record in records:
        wfdb.dl_database("mitdb", dl_dir=destination, records=[record])
            
            

def load_record(record_id: str, data_path: str = "data/raw"):
    """
    Load an ECG signal and its annotations from a specific record.
    """
    # Ensure path is absolute and correct regardless of where script is run
    full_path = os.path.abspath(os.path.join(data_path, record_id))
    record = wfdb.rdrecord(full_path)
    annotation = wfdb.rdann(full_path, 'atr')

    
    signal = record.p_signal #shape: (time, channels)
    ann_symbols = annotation.symbol
    ann_samples = annotation.sample # sample indices of annotations
    
    return {
      "record_id": record_id,
      "signal": signal, 
      "annotations": {
          "symbols": ann_symbols,
          "samples": ann_samples
      },
      "fs": record.fs #sampling frequency
      }


