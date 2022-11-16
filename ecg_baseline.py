import numpy as np
import scipy.signal as signal
# import for lms filter
import padasip as pa


class BaselineECG:
    def __init__(self, data, json_params):
        self.data = data
        self.fs = json_params['fs']
        self.N = json_params['n_sig']

    def moving_average(self, window_size):
        filtered = np.convolve(self.data, np.ones(window_size) / window_size, mode='valid')
        return filtered

    def butterworth_filter(self, Wn):
        b_butter, a_butter = signal.butter(self.N, Wn, self.fs, btype="high")
        return signal.filtfilt(b_butter, a_butter, self.data)

    def sav_goal_filter(self, window_length):
        return signal.savgol_filter(self.data, window_length, 3)

    def lms_filter(self):
        pass

# TODO: finish lms_filter, ask about missing parameter values, do main funct.
