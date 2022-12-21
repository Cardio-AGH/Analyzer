import numpy as np
import math as m
from scipy import interpolate, signal


class HRV1:
    def analyse(self, r_peaks_data, json_params):
        """
        Implementation of time and frequency HRV1 analysis
        Args:
            r_peaks (np.ndarray): sample numbers of R-waves
            json_params (dict)
        Returns:
            time_and_freq_parameters (dictionary)
        """

        self.fs = json_params['fs'] / 1000
        self.data = r_peaks_data / self.fs
        self.rr_intervals = self.rr_diff(self.data)

        time_analysis_parameters = self.__time_analysis(self.rr_intervals)
        freq_analysis_parametes = self.__frequency_analysis(self.data, self.fs)

        return {'time_parameters': time_analysis_parameters, 'frequency_parameters': freq_analysis_parametes}

    def __time_analysis(self, data):
        """
        Time analysis of HRV

        Args:
            data (np.array): sample numbers of R-waves
        Returns:
            dictionary: parameters of time analysis
                        - RR_mean,
                        - SDNN,
                        - RMSSD,
                        - NN50,
                        - pNN50
        """

        self.mean_rr = np.mean(data)
        self.sdnn = np.std(data)
        self.rmssd = m.sqrt(sum(self.rr_diff(data) ** 2) / (len(data) - 1))
        self.nn_50 = np.sum(np.abs(self.rr_diff(data) > 0.05))
        self.pnn_50 = 100 * self.nn_50 / len(data)

        time_parameters = {
            'mean_rr': self.mean_rr,
            'sdnn': self.sdnn,
            'rmssd': self.rmssd,
            'nn_50': self.nn_50,
            'pnn_50': self.pnn_50
        }
        return time_parameters

    def __frequency_analysis(self, data, fs):
        """
        Frequency analysis of HRV

        Args:
            data (np.array): sample numbers of R-waves
        Returns:
            dictionary: parameters of frequency analysis
                        -TP,
                        -HF,
                        -LF,
                        -VLF,
                        -ULF,
                        -LFHF
        """
        data = signal.resample(data, 100)
        rr_distance = self.rr_diff(data)
        data = data[:-1]
        ulf_band = [0, 0.003]
        vlf_band = [0.003, 0.04]
        lf_band = [0.04, 0.15]
        hf_band = [0.15, 0.4]

        interpolation = interpolate.interp1d(data, rr_distance, 'linear', bounds_error=False)
        timestamp_interpolation = np.arange(0, data[-1], 1 / float(fs))
        rr_interpolation = interpolation(timestamp_interpolation)
        rr_normalized = rr_interpolation - np.mean(rr_interpolation)
        f, pxx = signal.welch(x=rr_normalized, fs=fs, window='hann', nfft=4096)

        ulf_idx = np.logical_and(f >= ulf_band[0], f < ulf_band[1])
        vlf_idx = np.logical_and(f >= vlf_band[0], f < vlf_band[1])
        lf_idx = np.logical_and(f >= lf_band[0], f < lf_band[1])
        hf_idx = np.logical_and(f >= hf_band[0], f < hf_band[1])

        high_frequency = np.trapz(y=pxx[hf_idx], x=f[hf_idx])
        low_frequency = np.trapz(y=pxx[lf_idx], x=f[lf_idx])
        very_low_frequency = np.trapz(y=pxx[vlf_idx], x=f[vlf_idx])
        ultra_low_frequency = np.trapz(y=pxx[ulf_idx], x=f[ulf_idx])
        tp = ultra_low_frequency + very_low_frequency + low_frequency + high_frequency
        lfhf = low_frequency / high_frequency

        psd_chart = np.array([f, pxx])

        freqency_parameters = {
            'hf': high_frequency,
            'lf': low_frequency,
            'vlf': very_low_frequency,
            'ulf': ultra_low_frequency,
            'total_power': tp,
            'lfhf': lfhf,
            'psd_chart': psd_chart
        }
        return freqency_parameters

    def rr_diff(self, data):
        """
        Difference of length of two consecutive intervals
        """
        return np.diff(data)
