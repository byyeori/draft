import numpy as np
import pandas as pd

def load_data(imf_path, raw_path):
    imfs = pd.read_csv(imf_path).values.astype(np.float32)
    raw  = pd.read_csv(raw_path).values[:,0].astype(np.float32)
    return imfs, raw
