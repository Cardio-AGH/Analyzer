import numpy as np

class DFA:
    def main(self, r_peaks_data,json_params,windows=[4, 32]):
        """
         Args:
            r_peaks (np.ndarray): sample numbers of R-waves
            windows (list): A list with min/max values of windows range
        Returns:
            alpha_1_alpha_2_parameters (dict): Output parameteres for short-time windows and long-time windows
        """

        self.fs = json_params['fs'] / 1000
        self.data = r_peaks_data / self.fs
        self.rr_intervals = self.rr_diff(self.data)
        if self.rr_intervals.shape[0] < windows[1]:
            windows[1] = self.rr_intervals.shape[0]

        if windows[0] <= 8 and windows[1] <= 32:
            scales = np.concatenate(
                (
                    np.arange(windows[0], 8, 1),
                    np.arange(8, windows[1] + 1, 1),
                ),
                axis=0,
            )
        elif windows[0] <= 8 and windows[1] > 32:
            scales = np.concatenate(
                (
                    np.arange(windows[0], 8, 1),
                    np.arange(8, 32, 1),
                    np.arange(32, windows[1] + 1, 1)
                ),
                axis=0,
            )
        elif windows[0] > 8 and windows[1] > 32:
            scales = np.concatenate(
                (
                    np.arange(windows[0], 32, 1),
                    np.arange(32, windows[1] + 1, 1)
                ),
                axis=0,
            )
        else:
            scales = np.arange(windows[0], windows[1] + 1, 1)

        ix = int(np.where(scales == windows[0] * 2.5)[0] + 1)
        result = {
            "alpha_1": self.dfa(self.rr_intervals, scales[:ix]),
            "alpha_2": self.dfa(self.rr_intervals, scales[ix - 1:]),
        }

        return {'DFA':  result}

    def dfa(self, data: np.array, scales: np.array):
        """
        Args:
            data (np.array): Input signal after R-peaks detection
            scales (np.array): An array containing lim values of each window
        Returns:
            parameters (dict): A dictionary with output parameters of the module
        """
        integrated = self.integrate(data)
        flucts = self.fluctuations(integrated, scales)
        coeff = self.alpha(scales, flucts)
        fluctfit = 2 ** np.polyval(coeff, np.log2(scales))
        return {
            "Alpha": coeff[0],
            "Fluctuations": flucts,
            "Fluct_poly": fluctfit,
            "Windows": scales,

        }

    def integrate(self, data):
        """
        Integrates the input signal.

        Args:
            data (np.array): Input signal
        Returns:
            integrated (np.array): Integrated signal
        """
        return np.cumsum(data - np.mean(data))

    def fluctuations(self, integrated, scales):
        """
        Creates fluctuations array.

        Args:
            integrated (np.array): Integrated signal
            scales (np.array): Window size
        Returns:
            fluct (np.array): An array containing fluctuation values for each window size
        """

        # A helper funtion for calculating rms
        def _rms(integrated, scales):
            shape = (integrated.shape[0] // scales, scales)
            moving_window = np.lib.stride_tricks.as_strided(integrated, shape=shape)
            scale_ax = np.arange(scales)
            rms = np.zeros(moving_window.shape[0])
            for e, xcut in enumerate(moving_window):
                coeff = np.polyfit(scale_ax, xcut, 1)
                xfit = np.polyval(coeff, scale_ax)
                rms[e] = np.sqrt(np.mean((xcut - xfit) ** 2))
            return rms

        fluct = np.zeros(len(scales))
        for e, sc in enumerate(scales):
            fluct[e] = np.sqrt(np.mean(_rms(integrated, sc) ** 2))

        return fluct

    def alpha(self, scales, flucts):
        """
        Returns alpha coefficient.

        Args:
            scales (np.array): Window size
            fluct (np.array): An array containing fluctuation values for each window size
        Returns:
            coeff (np.array): An array containing alpha coefficient
            scales: tablica NumPy zawierająca rozmiary okien
        """
        return np.polyfit(np.log2(scales), np.log2(flucts), 1)

    def rr_diff(self, data):
        """
        Difference of length of two consecutive intervals.

        Args:
            data (np.array): Input signal
        Returns:
            rr_intervals (np.array): An array containing intervals between each peaks
        """
        return np.diff(data)
