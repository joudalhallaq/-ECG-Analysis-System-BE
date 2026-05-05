from django.urls import path
from .views import (
    upload_ecg,
    device_submit,
    list_ecg_records,
    analyze_ecg,
    get_ecg_result,
    delete_ecg,
    download_report,
)

urlpatterns = [
    path("upload/", upload_ecg),
    path("device-submit/", device_submit),
    path("records/", list_ecg_records),
    path("analyze/", analyze_ecg),
    path("result/<int:record_id>/", get_ecg_result),
    path("delete/<int:record_id>/", delete_ecg),
    path("report/<int:record_id>/", download_report),
]
