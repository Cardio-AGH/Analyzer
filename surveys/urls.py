from django.urls import path, include

from surveys.views import *

urlpatterns = [
    path(r'wav_details/<int:wav_pk>', WavFileDetailsView.as_view(), name='wav_details'),
    path(r'', WavlistView.as_view(), name='wav_list'),
    path(r'survey_fill/', WavFileAdd.as_view(), name='survey_fill'),
]
