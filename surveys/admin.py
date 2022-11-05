from django.contrib import admin

from surveys.models import Survey

# Register your models here.



@admin.register(Survey)
class SurveytAdmin(admin.ModelAdmin):
    pass