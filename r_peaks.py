import numpy as np
import scipy.signal as sig


class RPeaks:

    def __init__(self, data, json_params):
        self.data = data
        self.fs = json_params['fs']

    def pan_tompkins(self, data, window_size):

        """
        Pan Tompkins algorithm with 3 steps:
            - derivative filtering
            - squaring
            - moving window integration

        Args:
            data (np.ndarray): output of ecg_baseline
            window_size (int) : number of samples to average
        Returns:
            transformed_data (np.ndarray)
        """

        #Derivative filter apply

        derivative = np.convolve(data, [-1, 1], mode='same')

        #Squaring
        for i in range(len(derivative)):
            derivative[i] = derivative[i] ** 2

        #Moving window integration
        integrated_signal = np.zeros_like(derivative)
        cumulative_sum = derivative.cumsum()

        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)

        return integrated_signal



    def hilbert_transform(self, data, window_size):
        """
        Function that computes the analytic signal, using the Hilbert transform.

        Args:
            data (np.ndarray): output of pan_tompkins
            window_size (int) : number of samples to average
        Returns:
            transformed_data (np.ndarray)
        """

        signal = self.pan_tompkins(data, window_size)
        n = len(signal)
        transformed_data = sig.hilbert(signal, n, axis=-1)

        return transformed_data

    def r_peaks_detect(self, data, window_size):
        """
        Function that finds R peaks
        Args:
            data (np.ndarray): output of hilbert_transform
            window_size (int) : number of samples to average
        Returns:
            r_peaks(np.ndarray)
        """
        maximum = np.max(data)
        peaks, _ = sig.find_peaks(np.real(self.hilbert_transform(data, window_size)), height=maximum, distance=50,
                              prominence=None)

        signal = data
        num_peak = peaks.shape[0]
        r_peaks_list = list()

        for index in range(num_peak):
            i = peaks[index]
            i += 1
            cnt = i
            if cnt - 1 < 0:
                break

            if signal[cnt] < signal[cnt - 1]:
                while signal[cnt] < signal[cnt - 1]:
                    cnt -= 1

                    if cnt < 0:
                        break

            elif signal[cnt] > signal[cnt + 1]:
                while signal[cnt] < signal[cnt + 1]:
                    cnt += 1

                    if cnt < 0:
                        break

            r_peaks_list.append(cnt)

        r_peaks = np.asarray(r_peaks_list)

        return r_peaks