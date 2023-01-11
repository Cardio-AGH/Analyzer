from enum import unique
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.base import TemplateView
# Create your views here.
from django.views.generic import FormView
from surveys.forms import SurveyForm
from collections import Counter
from scipy.io import wavfile
from surveys.models import Survey
import numpy as np
from scipy.fft import fft, ifft, fftfreq

from ecg_baseline import BaselineECG
from r_peaks import RPeaks
from waves import Waves
from hrv1 import HRV1

from img_tool import matplotlib_to_base64, plot, plot_waves
# backend do matplotlib pozwalajacy na zapisywanie jako .png
import matplotlib
matplotlib.use('Agg')

class WavFileAdd(TemplateView, FormView):
    success_url = reverse_lazy('wav_list')
    template_name = "survey_form.html"

    def get_form_class(self):
        return SurveyForm

    def form_valid(self, form):
        form.instance.save()
        return super(WavFileAdd, self).form_valid(form)


    def get_context_data(self, **kwargs):
        context = super(WavFileAdd, self).get_context_data(**kwargs)
        return context

class WavFileDetailsView(TemplateView):
    template_name = "wav_details.html"

    def get_context_data(self, **kwargs):
        context = super(WavFileDetailsView, self).get_context_data(**kwargs)

        survey_answers = Survey.objects.get(pk = self.kwargs['wav_pk'])
        context['total_count'] = survey_answers

        samplerate, data = wavfile.read(survey_answers.wav_file.path)
        print(f'{samplerate:/^20}')
        flatten_data = [item for sublist in data for item in sublist if item % 10 == 0][0:1000]
        context['data'] = flatten_data
        context['data_labels'] = [round(i,2) for i in range(len(flatten_data))]
        context['mean'] = np.mean(flatten_data)
        context['std'] = np.std(flatten_data)



        # TODO Tutaj robimy przekształceia danych do kolejnych wykresów 

        #flatten_data = np.convolve(flatten_data, np.ones(6)/6, mode='valid')
        context['data_2'] = flatten_data#flatten_data[0:100]
        context['data_labels_2'] = [round(i,2) for i in range(len(flatten_data))]
        print(len(flatten_data), len([round(i,2) for i in range(len(flatten_data))]))
        return context



import numpy as np
import wfdb
from typing import Tuple

def load_data(path: str) -> Tuple[dict, np.array]:
  path = path.strip('.hea')
  data_dict = wfdb.rdheader(path).__dict__
  record = wfdb.rdrecord(path, physical=True, channels=[0])
  data_array = np.ravel(record.adc())
  fig = wfdb.plot_wfdb(record=record, figsize=(10,3), return_fig=True)
  print('dict')
  print(data_dict)
  print('array')
  print(data_array)
  return data_dict, data_array, fig



class WavlistView(TemplateView):
    template_name = "wav_list.html"

    def get_context_data(self, **kwargs):
        context = super(WavlistView, self).get_context_data(**kwargs)
        context['wavs'] = Survey.objects.all()


        data_dict, data, data_fig = load_data("twa00.hea")
        data_fig_base64 = matplotlib_to_base64(data_fig)
        baseline = BaselineECG(data, data_dict)
        algorytm1 = baseline.moving_average(10)
        fig1 = matplotlib_to_base64(plot(algorytm1))
        algorytm2 = baseline.butterworth_filter([5, 30])
        algorytm3 = baseline.sav_goal_filter(10)
        peaks = RPeaks(algorytm1, data_dict)
        algorytm4 = peaks.find_r_peaks(10, "pan tompkins")
        detect_waves = Waves(algorytm1, algorytm4)
        algorytm5 = detect_waves.main()
        fig5 = matplotlib_to_base64(plot_waves(data, algorytm1, algorytm5))

        context['data'] = data
        context['data_fig'] = data_fig_base64
        context['algorytm1'] = algorytm1
        context['fig1'] = fig1
        context['algorytm2'] = algorytm2
        context['algorytm3'] = algorytm3
        context['algorytm4'] = algorytm4
        context['algorytm5'] = algorytm5
        context['fig5'] = fig5

        return context
