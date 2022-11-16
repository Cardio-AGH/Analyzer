import os
import numpy as np
import json
import wfdb
from typing import Tuple

def load_data(path: str) -> Tuple[dict, np.array]:
  path = path.strip('.hea')
  data_dict = wfdb.rdheader(path).__dict__
  data_array = np.ravel(wfdb.rdrecord(path, physical=True, channels=[0]).adc())

  return data_dict, data_array