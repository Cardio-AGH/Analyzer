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
        #context['data_labels_2'] = data
        #context['data_2'] = data

        return context

class WavlistView(TemplateView):
    template_name = "wav_list.html"

    def get_context_data(self, **kwargs):
        context = super(WavlistView, self).get_context_data(**kwargs)
        context['wavs'] = Survey.objects.all()
        return context