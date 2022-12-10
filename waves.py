from r_peaks import RPeaks
from load_data import load_data
import numpy as np
from neurokit2 import signal
import padasip as pa
import neurokit as nk
import pandas as pd
import scipy


class Waves:
    """
    Class with methods which find characteristic points of ECG signal. It returns dictionary.
    """

    def compute_waves(self, ecg, rpeaks, fs=100):
        print("computing waves")
        analysis_fs = 3000
        ecg = scipy.signal.resample(ecg, int((ecg.shape[0] / fs) * analysis_fs))
        ecg = self.apply_gaussian(ecg, kernel_size=20)
        rpeaks = self.resample_indices(rpeaks, fs, analysis_fs)
        qpeaks = []
        speaks = []
        heartbeats = self.separate_beats(pd.Series(ecg), rpeaks)
        for i, heartbeat in enumerate(heartbeats):
            # Get index of R peaks
            rpeak = rpeaks[i]
            R = rpeaks[i] - heartbeat.index[0]
            # Q wave
            Q_index, Q = self.ecg_delineator_peak_Q(rpeak, heartbeat, R)
            qpeaks.append(Q_index)
            # S wave
            S_index, S = self.ecg_delineator_peak_S(rpeak, heartbeat)
            speaks.append(S_index)

        # dwt to delineate tp waves, onsets, offsets and qrs ontsets and offsets
        dwtmatr = self.dwt_compute_multiscales(ecg, 9)
        tpeaks, ppeaks = self.dwt_delineate_tp_peaks(ecg, rpeaks, dwtmatr, analysis_fs)
        qrs_onsets, qrs_offsets = self.dwt_delineate_qrs_bounds(rpeaks, dwtmatr, ppeaks, tpeaks, analysis_fs)
        ponsets, poffsets = self.dwt_delineate_tp_onsets_offsets(ppeaks, rpeaks, dwtmatr, analysis_fs)
        tonsets, toffsets = self.dwt_delineate_tp_onsets_offsets(
            tpeaks, rpeaks, dwtmatr, analysis_fs, onset_weight=0.6, duration_onset=0.6
        )
        return dict(
            ECG_P_Onsets=self.resample_indices(self.clean_na(ponsets), analysis_fs, fs),
            ECG_P_Offsets=self.resample_indices(self.clean_na(poffsets), analysis_fs, fs),
            ECG_QRS_Onsets=self.resample_indices(self.clean_na(qrs_onsets), analysis_fs, fs),
            ECG_QRS_Offsets=self.resample_indices(self.clean_na(qrs_offsets), analysis_fs, fs),
            ECG_T_Offsets=self.resample_indices(self.clean_na(toffsets), analysis_fs, fs),
        )

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    def apply_gaussian(self, x, kernel_size=10):
        assert len(x.shape) == 1
        kernel = self.gaussian(np.arange(-kernel_size * 2, kernel_size * 2), 0, kernel_size) / (
            2 * kernel_size
        )
        x_conv = np.convolve(x, kernel, mode='SAME')
        return x_conv

    def adjust_duration(self, rpeaks, sampling_rate, duration=None):
        average_rate = np.median(signal.signal_rate(peaks=rpeaks, sampling_rate=sampling_rate))
        return np.round(duration * (60 / average_rate), 3)

    def adjust_degree(self, rpeaks, sampling_rate):
        average_rate = np.median(signal.signal_rate(peaks=rpeaks, sampling_rate=sampling_rate))
        scale_factor = (sampling_rate / 250) / (average_rate / 60)
        return int(np.log2(scale_factor))

    def dwt_delineate_tp_peaks(self,
        ecg,
        rpeaks,
        dwtmatr,
        sampling_rate=250,
        qrs_width=0.13,
        p2r_duration=0.2,
        rt_duration=0.25,
        degree_tpeak=3,
        degree_ppeak=2,
        epsilon_T_weight=0.25,
        epsilon_P_weight=0.02,
    ):
        srch_bndry = int(0.5 * qrs_width * sampling_rate)
        degree_add = self.adjust_degree(rpeaks, sampling_rate)

        # sanitize search duration by HR
        p2r_duration = self.adjust_duration(rpeaks, sampling_rate, duration=p2r_duration)
        rt_duration = self.adjust_duration(rpeaks, sampling_rate, duration=rt_duration)

        tpeaks = []
        for rpeak_ in rpeaks:
            if np.isnan(rpeak_):
                tpeaks.append(np.nan)
                continue
            # search for T peaks from R peaks
            srch_idx_start = rpeak_ + srch_bndry
            srch_idx_end = rpeak_ + 2 * int(rt_duration * sampling_rate)
            dwt_local = dwtmatr[degree_tpeak + degree_add, srch_idx_start:srch_idx_end]
            height = epsilon_T_weight * np.sqrt(np.mean(np.square(dwt_local)))

            if len(dwt_local) == 0:
                tpeaks.append(np.nan)
                continue

            ecg_local = ecg[srch_idx_start:srch_idx_end]
            peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
            peaks = list(
                filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks)
            )  # pylint: disable=W0640
            if dwt_local[0] > 0:  # just append
                peaks = [0] + peaks

            # detect morphology
            candidate_peaks = []
            candidate_peaks_scores = []
            for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
                correct_sign = (
                    dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
                )  # pylint: disable=R1716
                if correct_sign:
                    idx_zero = (
                        signal.signal_zerocrossings(dwt_local[idx_peak : idx_peak_nxt + 1])[0]
                        + idx_peak
                    )
                    # This is the score assigned to each peak. The peak with the highest score will be
                    # selected.
                    score = ecg_local[idx_zero] - (
                        float(idx_zero) / sampling_rate - (rt_duration - 0.5 * qrs_width)
                    )
                    candidate_peaks.append(idx_zero)
                    candidate_peaks_scores.append(score)

            if not candidate_peaks:
                tpeaks.append(np.nan)
                continue

            tpeaks.append(
                candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start
            )

        ppeaks = []
        for rpeak in rpeaks:
            if np.isnan(rpeak):
                ppeaks.append(np.nan)
                continue

            # search for P peaks from Rpeaks
            srch_idx_start = rpeak - 2 * int(p2r_duration * sampling_rate)
            srch_idx_end = rpeak - srch_bndry
            dwt_local = dwtmatr[degree_ppeak + degree_add, srch_idx_start:srch_idx_end]
            height = epsilon_P_weight * np.sqrt(np.mean(np.square(dwt_local)))

            if len(dwt_local) == 0:
                ppeaks.append(np.nan)
                continue

            ecg_local = ecg[srch_idx_start:srch_idx_end]
            peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
            peaks = list(
                filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks)
            )
            if dwt_local[0] > 0:  # just append
                peaks = [0] + peaks

            # detect morphology
            candidate_peaks = []
            candidate_peaks_scores = []
            for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
                correct_sign = (
                    dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
                )  # pylint: disable=R1716
                if correct_sign:
                    idx_zero = (
                        signal.signal_zerocrossings(dwt_local[idx_peak : idx_peak_nxt + 1])[0]
                        + idx_peak
                    )
                    # This is the score assigned to each peak. The peak with the highest score will be
                    # selected.
                    score = ecg_local[idx_zero] - abs(
                        float(idx_zero) / sampling_rate - p2r_duration
                    )  # Minus p2r because of the srch_idx_start
                    candidate_peaks.append(idx_zero)
                    candidate_peaks_scores.append(score)

            if not candidate_peaks:
                ppeaks.append(np.nan)
                continue

            ppeaks.append(
                candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start
            )
        return tpeaks, ppeaks

    def dwt_delineate_tp_onsets_offsets(self,
        peaks,
        rpeaks,
        dwtmatr,
        sampling_rate=250,
        duration_onset=0.3,
        duration_offset=0.3,
        onset_weight=0.4,
        offset_weight=0.4,
        degree_onset=2,
        degree_offset=2,
    ):
        # sanitize search duration by HR
        duration_onset = self.adjust_duration(rpeaks, sampling_rate, duration=duration_onset)
        duration_offset = self.adjust_duration(rpeaks, sampling_rate, duration=duration_offset)
        degree = self.adjust_degree(rpeaks, sampling_rate)
        onsets = []
        offsets = []
        for i in range(len(peaks)):
            # look for onsets
            srch_idx_start = peaks[i] - int(duration_onset * sampling_rate)
            srch_idx_end = peaks[i]
            if srch_idx_start is np.nan or srch_idx_end is np.nan:
                onsets.append(np.nan)
                continue
            dwt_local = dwtmatr[degree_onset + degree, srch_idx_start:srch_idx_end]
            onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
            if len(onset_slope_peaks) == 0:
                onsets.append(np.nan)
                continue
            epsilon_onset = onset_weight * dwt_local[onset_slope_peaks[-1]]
            if not (dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
                onsets.append(np.nan)
                continue
            candidate_onsets = np.where(dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[
                0
            ]
            onsets.append(candidate_onsets[-1] + srch_idx_start)

            # # only for debugging
            # events_plot([candidate_onsets, onset_slope_peaks], dwt_local)
            # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
            # plt.show()

        for i in range(len(peaks)):  # pylint: disable=C0200
            # look for offset
            srch_idx_start = peaks[i]
            srch_idx_end = peaks[i] + int(duration_offset * sampling_rate)
            if srch_idx_start is np.nan or srch_idx_end is np.nan:
                offsets.append(np.nan)
                continue
            dwt_local = dwtmatr[degree_offset + degree, srch_idx_start:srch_idx_end]
            offset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
            if len(offset_slope_peaks) == 0:
                offsets.append(np.nan)
                continue
            epsilon_offset = -offset_weight * dwt_local[offset_slope_peaks[0]]
            if not (-dwt_local[offset_slope_peaks[0] :] < epsilon_offset).any():
                offsets.append(np.nan)
                continue
            candidate_offsets = (
                np.where(-dwt_local[offset_slope_peaks[0] :] < epsilon_offset)[0]
                + offset_slope_peaks[0]
            )
            offsets.append(candidate_offsets[0] + srch_idx_start)
        return onsets, offsets

    def dwt_delineate_qrs_bounds(self, rpeaks, dwtmatr, ppeaks, tpeaks, sampling_rate=250):
        degree = self.adjust_degree(rpeaks, sampling_rate)
        onsets = []
        for i in range(len(rpeaks)):
            # look for onsets
            srch_idx_start = ppeaks[i]
            srch_idx_end = rpeaks[i]
            if srch_idx_start is np.nan or srch_idx_end is np.nan:
                onsets.append(np.nan)
                continue
            dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
            onset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
            if len(onset_slope_peaks) == 0:
                onsets.append(np.nan)
                continue
            epsilon_onset = 0.5 * -dwt_local[onset_slope_peaks[-1]]
            if not (-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
                onsets.append(np.nan)
                continue
            candidate_onsets = np.where(
                -dwt_local[: onset_slope_peaks[-1]] < epsilon_onset
            )[0]
            onsets.append(candidate_onsets[-1] + srch_idx_start)

            # only for debugging

        offsets = []
        for i in range(len(rpeaks)):  # pylint: disable=C0200
            # look for offsets
            srch_idx_start = rpeaks[i]
            srch_idx_end = tpeaks[i]
            if srch_idx_start is np.nan or srch_idx_end is np.nan:
                offsets.append(np.nan)
                continue
            dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
            onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
            if len(onset_slope_peaks) == 0:
                offsets.append(np.nan)
                continue
            epsilon_offset = 0.5 * dwt_local[onset_slope_peaks[0]]
            if not (dwt_local[onset_slope_peaks[0] :] < epsilon_offset).any():
                offsets.append(np.nan)
                continue
            candidate_offsets = (
                np.where(dwt_local[onset_slope_peaks[0] :] < epsilon_offset)[0]
                + onset_slope_peaks[0]
            )
            offsets.append(candidate_offsets[0] + srch_idx_start)

        return onsets, offsets

    def dwt_compute_multiscales(self, ecg: np.ndarray, max_degree):
        """Return multiscales wavelet transforms."""

        def _apply_H_filter(signal_i, power=0):
            zeros = np.zeros(2 ** power - 1)
            timedelay = 2 ** power
            banks = np.r_[
                1.0 / 8,
                zeros,
                3.0 / 8,
                zeros,
                3.0 / 8,
                zeros,
                1.0 / 8,
            ]
            signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
            signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 2 steps
            return signal_f

        def _apply_G_filter(signal_i, power=0):
            zeros = np.zeros(2 ** power - 1)
            timedelay = 2 ** power
            banks = np.r_[2, zeros, -2]
            signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
            signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 1 step
            return signal_f

        dwtmatr = []
        intermediate_ret = np.array(ecg)
        for deg in range(max_degree):
            S_deg = _apply_G_filter(intermediate_ret, power=deg)
            T_deg = _apply_H_filter(intermediate_ret, power=deg)
            dwtmatr.append(S_deg)
            intermediate_ret = np.array(T_deg)
        dwtmatr = [
            arr[: len(ecg)] for arr in dwtmatr
        ]  # rescale transforms to the same length
        return np.array(dwtmatr)

    def ecg_delineator_peak_Q(self, rpeak, heartbeat, R):
        Q = signal.signal_findpeaks(-heartbeat, height_min=0.05 * (heartbeat.max() - heartbeat.min()))
        if not Q["Peaks"].size == 0:
            Q = Q["Peaks"][-1]
        else :
            Q = 0
        from_R = R - Q
        return rpeak - from_R, Q

    def ecg_delineator_peak_S(self, rpeak, heartbeat):
        S = signal.signal_findpeaks(-heartbeat, height_min=0.05 * (heartbeat.max() - heartbeat.min()))
        if not S["Peaks"].size == 0:
            S = S["Peaks" ][0]  # Select most left-hand side
        else:
            S = 0
        return rpeak + S, S

    def separate_beats(self, signal, rpeaks):
        heartbeats = []
        segments = rpeaks.copy()
        segments = np.convolve(segments, [1, -1], mode="SAME") // 2
        segments = rpeaks - segments
        segments = np.concatenate(([0], segments, [signal.shape[0]]))
        for i, _ in enumerate(segments[:-1]):
            is_r = np.logical_and(rpeaks > segments[i], rpeaks < segments[i + 1])
            if sum(is_r) == 1:
                heartbeats.append(signal[segments[i] : segments[i + 1]])
        return heartbeats

    def resample_indices(self, x, fs, target_fs):
        x = np.array(x)
        return ((x/fs)*target_fs).astype(int)

    def clean_na(self, x):
        x = np.array(x)
        return x[~np.isnan(x)]
