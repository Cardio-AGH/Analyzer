from django.db import models

# Create your models here.


class Survey(models.Model):
    sample_text = models.CharField(max_length=1024, blank=True, null=True)
    wav_file = models.FileField(upload_to='wav_files', blank=True, null=True)


