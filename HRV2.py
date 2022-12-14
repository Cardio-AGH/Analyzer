from ecg_baseline import BaselineECG
from r_peaks import RPeaks
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import statistics

#wczytanie danych i ograniczenie peaksów do 20 sztuk dla łatwosci odczytywania
json_params, data = load_data("101.hea")
baseline = BaselineECG(data, json_params).sav_goal_filter(4)
peaks = RPeaks(data, json_params).r_peaks_detect(baseline, 6)

class HRV2:
    def __init__(self, peaks):
        self.peaks = peaks
        self.rr_intervals = self.RR_diffs()

    def RR_diffs(self, difference=1):
        diffs = []
        for i in range(difference, len(self.peaks)):
            diffs.append(self.peaks[i]-peaks[i-difference])
        return diffs
        
    def RR_histogram(self):
        plt.hist(self.rr_intervals, bins=250)
        plt.show()

    def TINN(self):
        return np.std(self.rr_intervals)/np.mean(self.rr_intervals)

    def triangular_index(self):
        rr_max = np.max(peaks)
        rr_min = np.min(peaks)
        return (rr_max-rr_min)/(rr_max+rr_min)

    def SD1(self):
        return statistics.mean(self.rr_intervals)

    def SD2(self):
        return statistics.mean(self.RR_diffs(2))

    def poincare(self):
        fig, ax = plt.subplots()
        ax.scatter(self.rr_intervals[1:], self.rr_intervals[:-1], c="black")
        plt.show()

hrv2 = HRV2(peaks)
#print(hrv2.RR_diffs())
hrv2.RR_histogram()
hrv2.poincare()
print('TINN = ', hrv2.TINN())
print('Indeks trójkątny = ', hrv2.triangular_index())
print('SD1 = ', hrv2.SD1())
print('SD2 = ', hrv2.SD2())