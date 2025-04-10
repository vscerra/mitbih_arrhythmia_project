# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:41:44 2025

@author: vscerra
"""


import numpy as np
import os
import wfdb
import urllib.request
from scripts.utils import get_project_path

def download_mitbih(destination="data/raw/mitdb", records=None):
    import os
    import urllib.request

    if records is None:
        records = ["100", "101", "102", "103", "104"]

    destination = get_project_path(destination)
    os.makedirs(destination, exist_ok=True)

    base_url = "https://physionet.org/files/mitdb/1.0.0"
    extensions = ["hea", "dat", "atr"]

    for record in records:
        for ext in extensions:
            file_name = f"{record}.{ext}"
            url = f"{base_url}/{file_name}"
            local_path = os.path.join(destination, file_name)
            if not os.path.exists(local_path):
                print(f"Downloading {file_name} → {local_path}")
                urllib.request.urlretrieve(url, local_path)
            else:
                print(f"{file_name} already exists at {local_path}")


def load_record(record_id: str, data_path: str = "data/raw"):
    """
    Load an ECG signal and its annotations from a specific record.
    """
    # Ensure path is absolute and correct regardless of where script is run
    from scripts.utils import get_project_path

    full_path = get_project_path(data_path, "mitdb", record_id)
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


