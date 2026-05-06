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
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_CLASSES = None
_TRAIN_MEAN = None
_TRAIN_STD = None
_CONFIG = None


def load_assets():
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS
    global _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG

    if _INTERPRETER is None:
        _INTERPRETER = Interpreter(model_path=str(MODEL_PATH))
        _INTERPRETER.allocate_tensors()
        _INPUT_DETAILS = _INTERPRETER.get_input_details()
        _OUTPUT_DETAILS = _INTERPRETER.get_output_details()

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

    return (
        _INTERPRETER,
        _INPUT_DETAILS,
        _OUTPUT_DETAILS,
        _CLASSES,
        _TRAIN_MEAN,
        _TRAIN_STD,
        _CONFIG,
    )


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
        raise ValueError("No numeric values found in uploaded CSV file.")

    return values


def prepare_signal(file_path):
    (
        _,
        input_details,
        _,
        _,
        train_mean,
        train_std,
        config,
    ) = load_assets()

    values = read_numeric_csv(file_path)
    arr = np.array(values, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    input_shape = input_details[0]["shape"]

    expected_length = int(
        config.get(
            "signal_length",
            input_shape[1] if len(input_shape) > 1 else 1000
        )
    )

    expected_leads = int(
        config.get(
            "num_leads",
            input_shape[2] if len(input_shape) > 2 else 1
        )
    )

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
    (
        interpreter,
        input_details,
        output_details,
        classes,
        _,
        _,
        _,
    ) = load_assets()

    x = prepare_signal(file_path)

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    expected_shape = input_details[0]["shape"]

    if list(x.shape) != list(expected_shape):
        try:
            interpreter.resize_tensor_input(input_index, x.shape)
            interpreter.allocate_tensors()
        except Exception:
            raise ValueError(
                f"Model input shape mismatch. Expected {expected_shape}, got {x.shape}."
            )

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)

    probabilities = np.array(prediction[0], dtype=np.float32)

    class_index = int(np.argmax(probabilities))
    predicted_class = str(classes[class_index])
    confidence = float(probabilities[class_index])

    return predicted_class, confidence, probabilities.tolist()
