from django import forms
from datetime import datetime, date
import django
from django.contrib.admin.helpers import ActionForm

from surveys.models import Survey

class SurveyForm(forms.ModelForm):
    """
    Form to create and edit template instance. 
    """

    class Meta:
        model = Survey
        fields = ["sample_text", "sample_bool", 'wav_file']