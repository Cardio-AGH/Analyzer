import numpy as np
import scipy.signal as signal
import padasip as pa


def moving_average(data, window_size):
    filtered = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return filtered

def butterworth_filter(data, N, fs, Wn)
    b_butter, a_butter = signal.butter(N, Wn=Wn, fs=fs, btype="high")
    return signal.filtfilt(b_butter, a_butter, data)

def sav_goal_filter(data, window_length, polyorder, mode):
    return signal.savgol_filter(data, window_length, polyorder, mode)

def lms_filter()
