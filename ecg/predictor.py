import csv
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"

MODEL_PATH = MODEL_DIR / "model.tflite"
CLASSES_PATH = MODEL_DIR / "classes.json"
MEAN_PATH = MODEL_DIR / "train_mean.npy"
STD_PATH = MODEL_DIR / "train_std.npy"
CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"

_INTERPRETER = None
_CLASSES = None
_TRAIN_MEAN = None
_TRAIN_STD = None
_CONFIG = None


def load_assets():
    global _INTERPRETER, _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG

    if _INTERPRETER is None:
        _INTERPRETER = Interpreter(model_path=str(MODEL_PATH))
        _INTERPRETER.allocate_tensors()

    if _CLASSES is None:
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            _CLASSES = json.load(file)

    if _TRAIN_MEAN is None:
        _TRAIN_MEAN = np.load(MEAN_PATH)

    if _TRAIN_STD is None:
        _TRAIN_STD = np.load(STD_PATH)

    if _CONFIG is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                _CONFIG = json.load(file)
        else:
            _CONFIG = {}

    return _INTERPRETER, _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG


def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise ValueError("TFLite model input details could not be loaded.")

    if not output_details:
        raise ValueError("TFLite model output details could not be loaded.")

    return input_details, output_details


def read_numeric_csv(file_path):
    values = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)

        for row in reader:
            numeric_row = []

            for cell in row:
                try:
                    numeric_row.append(float(str(cell).strip()))
                except ValueError:
                    continue

            if numeric_row:
                values.append(numeric_row)

    if not values:
        raise ValueError(
            "No numeric values found in uploaded CSV file. Please upload an ECG signal CSV file."
        )

    return values


def prepare_signal(file_path, input_details, train_mean, train_std, config):
    values = read_numeric_csv(file_path)
    arr = np.array(values, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    input_shape = input_details[0].get("shape")

    if input_shape is None:
        raise ValueError("TFLite model input shape is missing.")

    input_shape = list(input_shape)

    default_length = 1000
    default_leads = 12

    if len(input_shape) >= 3:
        if input_shape[1] and input_shape[1] > 0:
            default_length = int(input_shape[1])
        if input_shape[2] and input_shape[2] > 0:
            default_leads = int(input_shape[2])

    expected_length = int(config.get("signal_length") or default_length)
    expected_leads = int(config.get("num_leads") or default_leads)

    if arr.shape[1] > expected_leads:
        arr = arr[:, :expected_leads]

    if arr.shape[1] < expected_leads:
        pad_width = expected_leads - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

    if arr.shape[0] > expected_length:
        arr = arr[:expected_length, :]

    if arr.shape[0] < expected_length:
        pad_length = expected_length - arr.shape[0]
        arr = np.pad(arr, ((0, pad_length), (0, 0)), mode="constant")

    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    try:
        arr = (arr - train_mean) / train_std
    except Exception:
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    return arr.astype(np.float32)


def predict_ecg_file(file_path):
    interpreter, classes, train_mean, train_std, config = load_assets()

    input_details, output_details = get_model_details(interpreter)

    x = prepare_signal(
        file_path=file_path,
        input_details=input_details,
        train_mean=train_mean,
        train_std=train_std,
        config=config,
    )

    input_index = input_details[0]["index"]

    expected_shape = list(input_details[0]["shape"])
    actual_shape = list(x.shape)

    if expected_shape != actual_shape:
        interpreter.resize_tensor_input(input_index, actual_shape)
        interpreter.allocate_tensors()

        input_details, output_details = get_model_details(interpreter)
        input_index = input_details[0]["index"]

    output_index = output_details[0]["index"]

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)

    if prediction is None:
        raise ValueError("TFLite model returned no prediction output.")

    prediction = np.array(prediction)

    if prediction.ndim == 2:
        probabilities = prediction[0].astype(np.float32)
    else:
        probabilities = prediction.astype(np.float32).reshape(-1)

    if len(probabilities) == 0:
        raise ValueError("TFLite model returned an empty prediction.")

    class_index = int(np.argmax(probabilities))

    if class_index >= len(classes):
        raise ValueError(
            f"Predicted class index {class_index} is outside classes list."
        )

    predicted_class = str(classes[class_index])
    confidence = float(probabilities[class_index])

    return predicted_class, confidence, probabilities.tolist()
