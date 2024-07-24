import pandas as pd
import zipfile
import os

def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        with zipfile.ZipFile(file_path, 'r') as z:
            for filename in z.namelist():
                with z.open(filename) as f:
                    data = pd.read_csv(f)
                    dataframes.append(data)
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data
