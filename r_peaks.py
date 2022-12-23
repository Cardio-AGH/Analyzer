import numpy as np
import scipy.signal as sig


class RPeaks:

    def __init__(self, data, json_params):
        self.data = data
        self.fs = json_params['fs']

    def pan_tompkins(self, window_size):

        """
        Pan Tompkins algorithm with 3 steps:
            - derivative filtering
            - squaring
            - moving window integration
        Args:
            window_size (int) : number of samples to average
        Returns:
            transformed_data (np.ndarray)
        """
        derivative = np.convolve(self.data, [-1, 1], mode='same')

        for i in range(len(derivative)):
            derivative[i] = derivative[i] ** 2

        integrated_signal = np.zeros_like(derivative)
        cumulative_sum = derivative.cumsum()

        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)

        return integrated_signal

    def hilbert_transform(self, window_size):

        """
        Algorithm of R peaks detection using Hilbert Transform
        - differentiation
        - window integration
        - hilbert transform
        Args:
            window_size (int) : number of samples to average
        Returns:
            transformed_data (np.ndarray)
        """

        dx1 = np.diff(self.data)
        dx2=dx1**2
        integrated_signal = np.zeros_like(dx2)
        cumulative_sum = dx2.cumsum()

        integrated_signal[window_size:] = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        integrated_signal[:window_size] = cumulative_sum[:window_size] / np.arange(1, window_size + 1)

        transformed_data = sig.hilbert(integrated_signal, len(self.data))

        return transformed_data

    def find_r_peaks(self, window_size, algorithm="pan tompkins"):

        '''
        The function finds local maxima in Pan Tompkins or Hilbert algorithm and corrects R peaks
        Args:
            window_size (int) : number of samples to average
            algorithm (str) : name of the algorithm
        Returns:
            r_peaks (np.ndarray)
        '''

        #Finding local maxima
        if algorithm == "pan tompkins":
            maximum = np.max(self.data)
            peaks, _ = sig.find_peaks(np.real(self.pan_tompkins(window_size)), height=maximum, distance=50,
                                      prominence=None)
        elif algorithm == "hilbert":
            maximum = np.max(self.data)
            peaks, _ = sig.find_peaks(np.real(self.hilbert_transform(window_size)), height=maximum, distance=50,
                                      prominence=None)
        #R peaks correction
        signal = self.data
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
