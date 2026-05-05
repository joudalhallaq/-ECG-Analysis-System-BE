import json
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def register_user(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    confirm_password = str(data.get("confirm_password", "")).strip()

    if not username or not email or not password or not confirm_password:
        return JsonResponse(
            {"error": "Username, email, password, and confirm password are required"},
            status=400,
        )

    if password != confirm_password:
        return JsonResponse(
            {"error": "Password and confirm password do not match"},
            status=400,
        )

    if User.objects.filter(username__iexact=username).exists():
        return JsonResponse(
            {"error": "This username is already taken"},
            status=400,
        )

    if User.objects.filter(email__iexact=email).exists():
        return JsonResponse(
            {"error": "This email is already registered"},
            status=400,
        )

    user = User.objects.create_user(
        username=username,
        email=email,
        password=password,
    )

    return JsonResponse(
        {
            "message": "User registered successfully",
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
        }
    )


@csrf_exempt
def login_user(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    username_or_email = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username_or_email or not password:
        return JsonResponse(
            {"error": "Username and password are required"},
            status=400,
        )

    user = (
        User.objects.filter(username__iexact=username_or_email).first()
        or User.objects.filter(email__iexact=username_or_email).first()
    )

    if user is None:
        return JsonResponse({"error": "Invalid credentials"}, status=400)

    if not user.check_password(password):
        return JsonResponse({"error": "Invalid credentials"}, status=400)

    return JsonResponse(
        {
            "message": "Login successful",
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
        }
    )
