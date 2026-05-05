import csv
import json
from io import BytesIO
from datetime import datetime
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.http import JsonResponse, FileResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .models import ECGRecord


def model_has_field(model_class, field_name):
    return any(field.name == field_name for field in model_class._meta.fields)


def get_file_field_name():
    if model_has_field(ECGRecord, "ecg_file"):
        return "ecg_file"
    if model_has_field(ECGRecord, "file"):
        return "file"
    return None


def get_record_file(record):
    file_field = get_file_field_name()
    if not file_field:
        return None
    return getattr(record, file_field, None)


def get_record_file_name(record):
    record_file = get_record_file(record)
    if record_file:
        try:
            return record_file.name
        except Exception:
            return str(record_file)
    return ""


def get_record_file_path(record):
    record_file = get_record_file(record)
    if record_file:
        try:
            return record_file.path
        except Exception:
            return None
    return None


def set_record_field_if_exists(record, field_name, value):
    if model_has_field(ECGRecord, field_name):
        setattr(record, field_name, value)



def extract_signal_values_from_csv(file_path, max_points=500):
    signal_values = []

    if not file_path:
        return signal_values

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            reader = csv.reader(file)

            for row in reader:
                for cell in row:
                    try:
                        value = float(str(cell).strip())
                        signal_values.append(value)

                        if len(signal_values) >= max_points:
                            return signal_values
                    except ValueError:
                        continue

    except Exception as error:
        print("Signal extraction error:", error)

    return signal_values
def build_short_explanation(predicted_condition):
    if predicted_condition == "Normal":
        return (
            "The ECG appears normal based on the AI analysis. "
            "No obvious abnormal pattern was detected in the uploaded ECG data."
        )

    if predicted_condition == "Not analyzed":
        return "This ECG record has not been analyzed yet."

    return (
        f"The AI model predicted that this ECG may be related to {predicted_condition}. "
        "This result should be reviewed by a qualified doctor."
    )


def build_detailed_explanation(predicted_condition, confidence):
    if confidence is not None:
        try:
            confidence_text = f"{round(float(confidence) * 100)}%"
        except Exception:
            confidence_text = "Not available"
    else:
        confidence_text = "Not available"

    return (
        f"The uploaded ECG data was received, prepared, and analyzed using the AI model. "
        f"The predicted result is {predicted_condition}, with a confidence level of "
        f"{confidence_text}. The confidence value represents how strongly the model "
        f"supports this prediction based on the available ECG input. "
        f"This explanation is intended to help non-specialized users understand the result, "
        f"but it must not be considered a final medical diagnosis."
    )


def medical_disclaimer():
    return (
        "This system supports ECG understanding and patient awareness. "
        "It does not replace professional medical diagnosis. "
        "Please consult a qualified doctor before making any medical decision."
    )


def xai_explanation_text():
    return (
        "Advanced explainable AI highlighting is not available yet. "
        "Future versions may highlight ECG regions or features that influenced the model decision."
    )


def split_text(text, max_chars=95):
    words = str(text).split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += f" {word}" if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def serialize_record(record):
    return {
        "id": record.id,
        "file_name": get_record_file_name(record),
        "uploaded_at": record.uploaded_at.isoformat() if hasattr(record, "uploaded_at") and record.uploaded_at else None,
        "predicted_condition": getattr(record, "predicted_condition", None),
        "confidence": getattr(record, "confidence", None),
        "short_explanation": getattr(record, "short_explanation", None) if model_has_field(ECGRecord, "short_explanation") else None,
    }


def run_ai_model_placeholder(record):
    """
    Runs the real trained ECG model only when Analyze is requested.
    TensorFlow is imported lazily to avoid loading it during server startup.
    """
    file_path = get_record_file_path(record)

    if not file_path:
        raise ValueError("ECG file path not found.")

    from .predictor import predict_ecg_file

    predicted_condition, confidence, probabilities = predict_ecg_file(file_path)

    return predicted_condition, confidence
@csrf_exempt
def upload_ecg(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    user_id = request.POST.get("user_id")
    uploaded_file = (
        request.FILES.get("ecg_file")
        or request.FILES.get("file")
        or request.FILES.get("ecg")
    )

    if not user_id or not uploaded_file:
        return JsonResponse({"error": "user_id and ecg_file are required"}, status=400)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({"error": "User not found"}, status=404)

    file_field = get_file_field_name()
    if not file_field:
        return JsonResponse(
            {"error": "No file field found in ECGRecord model. Expected ecg_file or file."},
            status=500
        )

    try:
        record = ECGRecord(user=user)
        setattr(record, file_field, uploaded_file)
        set_record_field_if_exists(record, "source_type", "uploaded_file")
        record.save()

        return JsonResponse({
            "message": "ECG uploaded successfully",
            "record_id": record.id,
            "file_name": get_record_file_name(record),
        })
    except Exception as error:
        return JsonResponse({"error": f"Upload failed: {str(error)}"}, status=500)


@csrf_exempt
def device_submit(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_id = data.get("user_id")
    ecg_data = data.get("ecg_data")

    if not user_id or not ecg_data:
        return JsonResponse({"error": "user_id and ecg_data are required"}, status=400)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return JsonResponse({"error": "User not found"}, status=404)

    file_field = get_file_field_name()
    if not file_field:
        return JsonResponse(
            {"error": "No file field found in ECGRecord model. Expected ecg_file or file."},
            status=500
        )

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"external_device_ecg_{timestamp}.csv"
        content = ContentFile(str(ecg_data).encode("utf-8"), name=file_name)

        record = ECGRecord(user=user)
        setattr(record, file_field, content)
        set_record_field_if_exists(record, "source_type", "external_device")
        record.save()

        return JsonResponse({
            "message": "ECG data received successfully from external device",
            "record_id": record.id,
            "file_name": get_record_file_name(record),
        })
    except Exception as error:
        return JsonResponse({"error": f"Device ECG submission failed: {str(error)}"}, status=500)


def list_ecg_records(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    user_id = request.GET.get("user_id")
    if not user_id:
        return JsonResponse({"error": "user_id is required"}, status=400)

    try:
        records = ECGRecord.objects.filter(user_id=user_id).order_by("-uploaded_at")
    except Exception:
        records = ECGRecord.objects.filter(user_id=user_id).order_by("-id")

    data = [serialize_record(record) for record in records]
    return JsonResponse(data, safe=False)


@csrf_exempt
def analyze_ecg(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    record_id = data.get("record_id")
    if not record_id:
        return JsonResponse({"error": "record_id is required"}, status=400)

    record = get_object_or_404(ECGRecord, id=record_id)

    file_path = get_record_file_path(record)
    signal_values = extract_signal_values_from_csv(file_path)

    try:
        predicted_condition, confidence = run_ai_model_placeholder(record)

        short_explanation = build_short_explanation(predicted_condition)
        detailed_explanation = build_detailed_explanation(predicted_condition, confidence)

        set_record_field_if_exists(record, "predicted_condition", predicted_condition)
        set_record_field_if_exists(record, "confidence", confidence)
        set_record_field_if_exists(record, "short_explanation", short_explanation)
        set_record_field_if_exists(record, "detailed_explanation", detailed_explanation)
        set_record_field_if_exists(record, "xai_explanation", xai_explanation_text())

        record.save()

        return JsonResponse({
            "message": "ECG analyzed successfully",
            "record_id": record.id,
            "predicted_condition": predicted_condition,
            "confidence": confidence,
            "short_explanation": short_explanation,
            "detailed_explanation": detailed_explanation,
            "xai_explanation": xai_explanation_text(),
            "signal_values": signal_values,
            "disclaimer": medical_disclaimer(),
        })
    except Exception as error:
        return JsonResponse({"error": f"AI model failed to analyze ECG: {str(error)}"}, status=500)


def get_ecg_result(request, record_id):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    record = get_object_or_404(ECGRecord, id=record_id)

    predicted_condition = getattr(record, "predicted_condition", None) or "Not analyzed"
    confidence = getattr(record, "confidence", None)

    short_explanation = getattr(record, "short_explanation", None) if model_has_field(ECGRecord, "short_explanation") else None
    detailed_explanation = getattr(record, "detailed_explanation", None) if model_has_field(ECGRecord, "detailed_explanation") else None

    if not short_explanation:
        short_explanation = build_short_explanation(predicted_condition)

    if not detailed_explanation:
        detailed_explanation = build_detailed_explanation(predicted_condition, confidence)

    file_path = get_record_file_path(record)
    signal_values = extract_signal_values_from_csv(file_path)

    return JsonResponse({
        "record_id": record.id,
        "id": record.id,
        "file_name": get_record_file_name(record),
        "uploaded_at": record.uploaded_at.isoformat() if hasattr(record, "uploaded_at") and record.uploaded_at else None,
        "predicted_condition": predicted_condition,
        "confidence": confidence,
        "short_explanation": short_explanation,
        "detailed_explanation": detailed_explanation,
        "xai_explanation": xai_explanation_text(),
        "signal_values": signal_values,
        "disclaimer": medical_disclaimer(),
    })


@csrf_exempt
def delete_ecg(request, record_id):
    if request.method not in ["DELETE", "POST"]:
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        record = ECGRecord.objects.get(id=record_id)
    except ECGRecord.DoesNotExist:
        return JsonResponse({"error": "Record not found"}, status=404)

    try:
        record_file = get_record_file(record)
        if record_file:
            try:
                record_file.delete(save=False)
            except Exception:
                pass

        record.delete()
        return JsonResponse({"message": "ECG record deleted successfully"})
    except Exception as error:
        return JsonResponse({"error": f"Delete failed: {str(error)}"}, status=500)


def download_report(request, record_id):
    record = get_object_or_404(ECGRecord, id=record_id)

    predicted_condition = getattr(record, "predicted_condition", None) or "Not analyzed"
    confidence = getattr(record, "confidence", None)

    if confidence is not None:
        try:
            confidence_text = f"{round(float(confidence) * 100)}%"
        except Exception:
            confidence_text = "Not available"
    else:
        confidence_text = "Not available"

    short_explanation = build_short_explanation(predicted_condition)
    detailed_explanation = build_detailed_explanation(predicted_condition, confidence)
    disclaimer = medical_disclaimer()

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 60

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "ECG Analysis Report")

    y -= 35
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Record ID: {record.id}")

    y -= 22
    pdf.drawString(50, y, f"File Name: {get_record_file_name(record) or 'N/A'}")

    y -= 22
    uploaded_at = record.uploaded_at if hasattr(record, "uploaded_at") else "N/A"
    pdf.drawString(50, y, f"Uploaded At: {uploaded_at}")

    y -= 35
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Predicted Condition:")

    y -= 20
    pdf.setFont("Helvetica", 11)
    pdf.drawString(70, y, predicted_condition)

    y -= 30
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Confidence:")

    y -= 20
    pdf.setFont("Helvetica", 11)
    pdf.drawString(70, y, confidence_text)

    y -= 35
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Short Explanation:")

    y -= 20
    pdf.setFont("Helvetica", 10)
    for line in split_text(short_explanation, 95):
        pdf.drawString(70, y, line)
        y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Detailed Explanation:")

    y -= 20
    pdf.setFont("Helvetica", 10)
    for line in split_text(detailed_explanation, 95):
        if y < 90:
            pdf.showPage()
            y = height - 60
            pdf.setFont("Helvetica", 10)

        pdf.drawString(70, y, line)
        y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Explainable Interpretation:")

    y -= 20
    pdf.setFont("Helvetica", 10)
    for line in split_text(xai_explanation_text(), 95):
        if y < 90:
            pdf.showPage()
            y = height - 60
            pdf.setFont("Helvetica", 10)

        pdf.drawString(70, y, line)
        y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Medical Disclaimer:")

    y -= 20
    pdf.setFont("Helvetica", 10)
    for line in split_text(disclaimer, 95):
        if y < 90:
            pdf.showPage()
            y = height - 60
            pdf.setFont("Helvetica", 10)

        pdf.drawString(70, y, line)
        y -= 16

    pdf.showPage()
    pdf.save()

    buffer.seek(0)

    return FileResponse(
        buffer,
        as_attachment=True,
        filename=f"ecg_report_{record.id}.pdf",
        content_type="application/pdf",
    )
