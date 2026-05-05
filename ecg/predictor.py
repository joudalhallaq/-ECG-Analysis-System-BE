import csv
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"

MODEL_PATH = MODEL_DIR / "model.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"
MEAN_PATH = MODEL_DIR / "train_mean.npy"
STD_PATH = MODEL_DIR / "train_std.npy"
CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"

_MODEL = None
_CLASSES = None
_TRAIN_MEAN = None
_TRAIN_STD = None
_CONFIG = None


def load_assets():
    global _MODEL, _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG

    if _MODEL is None:
        import tensorflow as tf
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

    return _MODEL, _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG


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
    _, _, train_mean, train_std, config = load_assets()

    values = read_numeric_csv(file_path)
    arr = np.array(values, dtype=np.float32)

    expected_length = int(config.get("signal_length", 1000))
    expected_leads = int(config.get("num_leads", 12))

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

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

    arr = np.expand_dims(arr, axis=0)

    arr = (arr - train_mean) / train_std

    return arr


def predict_ecg_file(file_path):
    model, classes, _, _, _ = load_assets()

    x = prepare_signal(file_path)

    prediction = model.predict(x, verbose=0)

    if isinstance(prediction, list):
        prediction = prediction[0]

    probabilities = np.array(prediction[0], dtype=np.float32)

    class_index = int(np.argmax(probabilities))
    predicted_class = str(classes[class_index])
    confidence = float(probabilities[class_index])

    return predicted_class, confidence, probabilities.tolist()
