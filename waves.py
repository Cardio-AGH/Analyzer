import numpy as np
import scipy


class Waves:
    """
    Class with methods which find characteristic points of ECG signal. It returns dictionary.
    """

    def __init__(self, data, rpeaks):
        self.data = data
        self.rpeaks = rpeaks

    def waves_gradient(self):
        first_deriv = np.gradient(self.data)  # 1st derivative
        second_deriv = np.gradient(self.data)  # 2nd derivative
        return first_deriv, second_deriv

    def find_qrs_onsets(self, iter, deriv_2):
        _, _, _, quarter4 = np.array_split(deriv_2[self.rpeaks[iter]:self.rpeaks[iter + 1]],4)  # first derivative from ech_baseline split in 4
        try:
            QRSon = quarter4
            QRSon = np.where(QRSon[:np.argmin(QRSon)] < 0)[0][-2]
            QRSon = self.rpeaks[iter + 1] - len(quarter4) + QRSon
        except:
            return
        return int(QRSon)

    def find_qrs_offsets(self, iter, deriv_2):
        quarter1, _, _, _ = np.array_split(deriv_2[self.rpeaks[iter]:self.rpeaks[iter + 1]],4)  # first derivative from ech_baseline split in 4
        QRSoff = quarter1
        try:
            half1, half2 = np.array_split(self.data[self.rpeaks[iter]:self.rpeaks[iter + 1]], 2)
            R1 = self.rpeaks[iter]

            S, _ = scipy.signal.find_peaks(-half1)
            if S.shape[0] != 0:
                S = np.min(S)
                S = R1 + S
                QRSoff = np.where(QRSoff[np.argmax(QRSoff):] < 0)[0][1]
                QRSoff = S + QRSoff
            else:
                return
        except:
            return
        return int(QRSoff)

    def find_t_offsets(self, iter, deriv_1, deriv_2, QRSoff):
        half1, _ = np.array_split(self.data[self.rpeaks[iter]:self.rpeaks[iter + 1]], 2)  # ecg_baseline between 2 R peaks split in half
        try:
            region = self.data[QRSoff:QRSoff + len(half1)]
            T, _ = scipy.signal.find_peaks(region)
            if T.shape[0] != 0:
                Tmax_index = np.argmax(region[T])
                T = T[Tmax_index]
                T = QRSoff + T
            else:
                return
        except:
            return
        quarter1, _, _, _ = np.array_split(deriv_2[self.rpeaks[iter]:self.rpeaks[iter + 1]],4)  # first derivative from ech_baseline split in 4
        try:
            dif_max, _ = scipy.signal.find_peaks(-deriv_1[T:T + len(quarter1)])
            if dif_max.shape[0] != 0:
                dif_max = np.min(dif_max)
                Toffset = T + 3 * dif_max
            else:
                return
        except:
            return
        return int(Toffset)

    def find_p_peaks(self, iter):
        gauss_half1, gauss_half2 = np.array_split(self.data[self.rpeaks[iter]:self.rpeaks[iter + 1]], 2)
        try:
            P, _ = scipy.signal.find_peaks(gauss_half2)
            if P.shape[0] != 0:
                Pmax_index = np.argmax(gauss_half2[P])
                P = P[Pmax_index]
                P = self.rpeaks[iter] + len(gauss_half1) + P
            else:
                return
        except:
            return
        return int(P)

    def find_p_onsets(self, iter, deriv_1, P):
        half1, _ = np.array_split(self.data[self.rpeaks[iter]:self.rpeaks[iter + 1]], 2)  # ecg_baseline between 2 R peaks split in half
        try:
            dif_max, _ = scipy.signal.find_peaks(deriv_1[len(half1):P])
            if dif_max.shape[0] != 0:
                dif_max = np.max(dif_max)
                Ponset = P - 2 * (P - len(half1) - dif_max)
            else:
                return
        except:
            return
        return int(Ponset)

    def find_p_offsets(self, deriv_1, P, QRSon):
        try:
            dif_max, _ = scipy.signal.find_peaks(-deriv_1[P:QRSon])
            if dif_max.shape[0] != 0:
                dif_max = np.min(dif_max)
                Poffset = P + 2 * dif_max
            else:
                return
        except:
            return
        return int(Poffset)

    def delete_empty(self, lista):
        lista_result = []
        for iter in lista:
            if iter is not None:
                lista_result.append(iter)

        return np.array(lista_result)

    def main(self):
        ECG_QRS_Onsets = []
        ECG_QRS_Offsets = []
        ECG_P_Onsets = []
        ECG_P_Offsets = []
        ECG_T_Offsets = []

        deriv_1, deriv_2 = self.waves_gradient()

        for iter, _ in enumerate(self.rpeaks[:-1]):
            QRSoff = self.find_qrs_offsets(iter, deriv_2)
            QRSon = self.find_qrs_onsets(iter, deriv_2)
            Toff = self.find_t_offsets(iter, deriv_1, deriv_2, QRSoff)
            P = self.find_p_peaks(iter)
            Pon = self.find_p_onsets(iter, deriv_1, P)
            Poff = self.find_p_offsets(deriv_1, P, QRSon)

            ECG_QRS_Offsets.append(QRSoff)
            ECG_QRS_Onsets.append(QRSon)
            ECG_P_Onsets.append(Pon)
            ECG_P_Offsets.append(Poff)
            ECG_T_Offsets.append(Toff)
            offset = -2

        return dict(
            ECG_QRS_Offsets=self.delete_empty(ECG_QRS_Offsets) + offset,
            ECG_T_Offsets=self.delete_empty(ECG_T_Offsets) + offset,
            ECG_QRS_Onsets=self.delete_empty(ECG_QRS_Onsets) + offset,
            ECG_P_Onsets=self.delete_empty(ECG_P_Onsets) + offset,
            ECG_P_Offsets=self.delete_empty(ECG_P_Offsets) + offset
        )
