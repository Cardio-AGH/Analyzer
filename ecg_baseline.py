import numpy as np
import scipy.signal as signal
import padasip as pa


class BaselineECG:
    """
    Class with implementation of methods needed to perform filtration on an ECG signal.
    """
    def __init__(self, data, json_params):
        self.data = data
        self.fs = json_params['fs']
        self.N = json_params['n_sig']

    def moving_average(self, window_size):
        """Function filtering input data with moving average.

        Args:
            window_size: int value with size of the window used in filtration.

        Returns:
            filtered: np.ndarray data with filtered values.
        """
        filtered = np.convolve(self.data, np.ones(window_size) / window_size, mode='valid')
        return filtered

    def butterworth_filter(self, Wn):
        """Function filtering input data with butterworth filter.

        Args:
            Wn: array_like with critical frequency.

        Returns:
            np.ndarray data with filtered values.
        """
        b_butter, a_butter = signal.butter(self.N, Wn, 'bandpass', fs=self.fs)
        return signal.filtfilt(b_butter, a_butter, self.data, )

    def sav_goal_filter(self, window_length):
        """Function filtering input data with Savitzky-Golay filter.

        Args:
            window_length: int value with size of the window used in filtration.

        Returns:
            np.ndarray data with filtered values.
        """
        return signal.savgol_filter(self.data, window_length, 3)

    def lms_filter(self, filtered_data, n):
        """Function filtering input data with LMS filter. To use this filter the input data should be
        filtered by simple low-pass filter. (https://matousc89.github.io/padasip/sources/filters/lms.html)

        Args:
            filtered_data: input data filtered by simple low-pass filter,
            n: filter length,
            target: function e.g. d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3]

        Returns:
            np.ndarray data with filtered values.
        """
        # filter = pa.filters.FilterLMS(n, mu=0.1)
        # pred = filter.predict(self.data)
        # # pred = pred[:len(filtered_data)]
        # # print(len(pred), len(filtered_data))
        # y, _, _ = filter.run(pred, filtered_data)
        x = filtered_data # input matrix
        v = np.random.normal(0, 0.1, n)  # noise
        d = x + v
        f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
        y, e, w = f.run(d, x)
        return y