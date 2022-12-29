import load_data
import ecg_baseline
from r_peaks import RPeaks
from hrv1 import HRV1
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

#ładowanie danych
json_params, data = load_data.load_data(r'twa00.hea')


#algorytmy Baseline i R_peaks
#stworzenie obiektu klasy BaselineECG
ecg = ecg_baseline.BaselineECG(data, json_params)
#przefiltrowanie danych z obiektu BaslineECG
ecg_filtered = ecg.moving_average(4)
#stworzenie obiektu klas RPeaks, przekazanie danych z ecg baseline
r = RPeaks(ecg_filtered, json_params)
#detekcja pików
der = r.find_r_peaks(6)


hrv1 = HRV1()
hrv1_data = hrv1.analyse(der, json_params)

print(f'Time parameters: {hrv1_data["time_parameters"]}')
print(f'Frequency parameters: {hrv1_data["frequency_parameters"]}')



#wykres, ale Fabian nie kazał go robić, to taki dodatek
hrv1_frequency_parameters = hrv1_data.get('frequency_parameters')
hrv1 = hrv1_frequency_parameters.get('psd_chart')
f = hrv1[0]
pxx = hrv1[1]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=f,
    y=pxx,
    line=dict(color="#008CFF"),
    name="hrv1"
))

fig.update_layout(
    title="Wykres hrv1",
    xaxis_title="Częstotliwość [Hz]",
    yaxis_title="Moc widma [ms^2]",
    showlegend=False,
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    yaxis_gridcolor="#9C9C9C",
    xaxis_gridcolor="#9C9C9C",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#9C9C9C")
)

fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fig.show()
