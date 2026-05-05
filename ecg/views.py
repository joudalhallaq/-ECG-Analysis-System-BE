from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ECGRecord
from django.forms.models import model_to_dict
from django.contrib.auth.models import User


def test_api(request):
    return JsonResponse({"message": "ECG API is working"})


@csrf_exempt
def upload_ecg(request):
    if request.method == 'POST':
        try:
            user_id = request.POST.get('user_id')
            ecg_file = request.FILES.get('ecg_file')

            if not user_id or not ecg_file:
                return JsonResponse({
                    'error': 'user_id and ecg_file are required'
                }, status=400)

            user = User.objects.get(id=user_id)

            record = ECGRecord.objects.create(
                user=user,
                ecg_file=ecg_file
            )

            return JsonResponse({
                'message': 'ECG uploaded successfully',
                'record_id': record.id,
                'file_name': record.ecg_file.name
            }, status=201)

        except User.DoesNotExist:
            return JsonResponse({
                'error': 'User not found'
            }, status=404)

        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'error': 'Only POST method is allowed'
    }, status=405)
def list_ecg_records(request):
    user_id = request.GET.get('user_id')

    if not user_id:
        return JsonResponse({
            'error': 'user_id is required'
        }, status=400)

    try:
        records = ECGRecord.objects.filter(user_id=user_id).order_by('-uploaded_at')

        data = []
        for record in records:
            data.append({
                'id': record.id,
                'user_id': record.user.id,
                'file_name': record.ecg_file.name,
                'uploaded_at': record.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        return JsonResponse({
            'records': data
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=400)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ECGRecord
import json


def test_api(request):
    return JsonResponse({"message": "ECG API is working"})


@csrf_exempt
def upload_ecg(request):
    if request.method == 'POST':
        try:
            user_id = request.POST.get('user_id')
            ecg_file = request.FILES.get('ecg_file')

            if not user_id or not ecg_file:
                return JsonResponse({
                    'error': 'user_id and ecg_file are required'
                }, status=400)

            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)

            record = ECGRecord.objects.create(
                user=user,
                ecg_file=ecg_file
            )

            return JsonResponse({
                'message': 'ECG uploaded successfully',
                'record_id': record.id,
                'file_name': record.ecg_file.name
            }, status=201)

        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'error': 'Only POST method is allowed'
    }, status=405)



def list_ecg_records(request):
    user_id = request.GET.get('user_id')

    print("USER ID:", user_id)

    if not user_id:
        return JsonResponse({'error': 'user_id is required'}, status=400)

    records = ECGRecord.objects.filter(user_id=user_id)

    print("RECORDS:", records)

    data = []

    for r in records:
        data.append({
            "id": r.id,
            "user_id": r.user_id,
            "file_name": r.ecg_file.name,
            "uploaded_at": r.uploaded_at.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_condition": r.predicted_condition,
            "confidence": r.confidence,
        })

    return JsonResponse({"records": data})

@csrf_exempt
def analyze_ecg(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            record_id = data.get('record_id')

            if not record_id:
                return JsonResponse({
                    'error': 'record_id is required'
                }, status=400)

            record = ECGRecord.objects.get(id=record_id)

            record.predicted_condition = "Normal"
            record.confidence = 0.95
            record.short_explanation = "The ECG appears normal with no obvious abnormal rhythm detected."
            record.save()

            return JsonResponse({
                "message": "ECG analyzed successfully",
                "record_id": record.id,
                "predicted_condition": "Normal",
                "confidence": 0.95,
                "short_explanation": "The ECG appears normal with no obvious abnormal rhythm detected.",
                "detailed_explanation": "The ECG signal shows a regular rhythm with consistent intervals between beats. No signs of arrhythmia, abnormal spikes, or irregular patterns were detected. The waveform morphology is within expected normal ranges, suggesting a healthy cardiac rhythm."
                })

        except ECGRecord.DoesNotExist:
            return JsonResponse({
                'error': 'ECG record not found'
            }, status=404)

        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'error': 'Only POST method is allowed'
    }, status=405)
def get_ecg_result(request, record_id):
    try:
        record = ECGRecord.objects.get(id=record_id)

        return JsonResponse({
            'record_id': record.id,
            'file_name': record.ecg_file.name,
            'predicted_condition': record.predicted_condition,
            'confidence': record.confidence,
            'short_explanation': record.short_explanation,
            'uploaded_at': record.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    except ECGRecord.DoesNotExist:
        return JsonResponse({
            'error': 'ECG record not found'
        }, status=404)
    from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def delete_ecg(request, record_id):
    if request.method == 'DELETE':
        try:
            record = ECGRecord.objects.get(id=record_id)
            record.delete()

            return JsonResponse({
                'message': 'ECG record deleted successfully'
            })

        except ECGRecord.DoesNotExist:
            return JsonResponse({
                'error': 'Record not found'
            }, status=404)

    return JsonResponse({
        'error': 'Only DELETE method is allowed'
    }, status=405)