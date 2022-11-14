# Generated by Django 3.2.3 on 2022-05-23 21:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('surveys', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='survey',
            name='wav_file',
            field=models.FileField(db_index=True, default=2, upload_to='wav_files'),
            preserve_default=False,
        ),
    ]