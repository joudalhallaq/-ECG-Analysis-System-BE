from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            username = data.get('username')
            email = data.get('email')
            password = data.get('password')

            if not username or not email or not password:
                return JsonResponse({
                    'error': 'Username, email, and password are required'
                }, status=400)

            if User.objects.filter(username=username).exists():
                return JsonResponse({
                    'error': 'Username already exists'
                }, status=400)

            if User.objects.filter(email=email).exists():
                return JsonResponse({
                    'error': 'Email already exists'
                }, status=400)

            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )

            return JsonResponse({
                'message': 'User registered successfully',
                'user_id': user.id,
                'username': user.username
            }, status=201)

        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'error': 'Only POST method is allowed'
    }, status=405)
from django.contrib.auth import authenticate

@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            username = data.get('username')
            password = data.get('password')

            user = authenticate(username=username, password=password)

            if user is not None:
                return JsonResponse({
                    'message': 'Login successful',
                    'user_id': user.id,
                    'username': user.username
                })
            else:
                return JsonResponse({
                    'error': 'Invalid credentials'
                }, status=400)

        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'error': 'Only POST method is allowed'
    }, status=405)