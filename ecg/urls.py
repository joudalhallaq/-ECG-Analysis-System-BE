from django.urls import path
from .views import (
    test_api,
    upload_ecg,
    list_ecg_records,
    analyze_ecg,
    get_ecg_result,
    delete_ecg,
    download_report,
    device_submit,
)

urlpatterns = [
    path("test/", test_api),
    path("upload/", upload_ecg),
    path("records/", list_ecg_records),
    path("analyze/", analyze_ecg),
    path("result/<int:record_id>/", get_ecg_result),
    path("delete/<int:record_id>/", delete_ecg),
    path("report/<int:record_id>/", download_report),
    path("device-submit/", device_submit),
]
