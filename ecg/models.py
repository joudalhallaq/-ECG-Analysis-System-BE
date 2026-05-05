from django.db import models
from django.contrib.auth.models import User


class ECGRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ecg_file = models.FileField(upload_to='ecg_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    predicted_condition = models.CharField(max_length=100, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    short_explanation = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"ECG Record {self.id} - {self.user.username}"