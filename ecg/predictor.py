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


_CLASSES = None
_TRAIN_MEAN = None
_TRAIN_STD = None
_CONFIG = None


def load_metadata():
    global _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG

    if _CLASSES is None:
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            loaded_classes = json.load(file)

        if isinstance(loaded_classes, dict):
            _CLASSES = list(loaded_classes.values())
        else:
            _CLASSES = loaded_classes

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

    return _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG


def create_interpreter():
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise ValueError("TFLite model input details are empty.")

    if not output_details:
        raise ValueError("TFLite model output details are empty.")

    return interpreter, input_details, output_details


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


def get_expected_shape(input_details, config):
    shape = list(input_details[0].get("shape", []))

    expected_length = 1000
    expected_leads = 12

    if len(shape) >= 3:
        if int(shape[1]) > 0:
            expected_length = int(shape[1])
        if int(shape[2]) > 0:
            expected_leads = int(shape[2])

    expected_length = int(
        config.get("signal_length")
        or config.get("expected_length")
        or config.get("sequence_length")
        or expected_length
    )

    expected_leads = int(
        config.get("num_leads")
        or config.get("n_leads")
        or config.get("leads")
        or expected_leads
    )

    return expected_length, expected_leads


def prepare_signal(file_path, input_details, train_mean, train_std, config):
    values = read_numeric_csv(file_path)
    arr = np.array(values, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    expected_length, expected_leads = get_expected_shape(input_details, config)

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


def apply_input_dtype_and_quantization(x, input_detail):
    input_dtype = input_detail.get("dtype", np.float32)

    if input_dtype == np.float32:
        return x.astype(np.float32)

    quantization = input_detail.get("quantization", None)

    if quantization:
        scale, zero_point = quantization

        if scale and scale > 0:
            x = x / scale + zero_point

    return x.astype(input_dtype)


def dequantize_output(output, output_detail):
    output = np.array(output)

    if output.dtype == np.float32:
        return output.astype(np.float32)

    quantization = output_detail.get("quantization", None)

    if quantization:
        scale, zero_point = quantization

        if scale and scale > 0:
            output = (output.astype(np.float32) - zero_point) * scale

    return output.astype(np.float32)


def predict_ecg_file(file_path):
    classes, train_mean, train_std, config = load_metadata()

    interpreter, input_details, output_details = create_interpreter()

    x = prepare_signal(
        file_path=file_path,
        input_details=input_details,
        train_mean=train_mean,
        train_std=train_std,
        config=config,
    )

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    expected_shape = list(input_details[0]["shape"])
    actual_shape = list(x.shape)

    if expected_shape != actual_shape:
        interpreter.resize_tensor_input(input_index, actual_shape, strict=False)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

    x = apply_input_dtype_and_quantization(x, input_details[0])

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    prediction = dequantize_output(prediction, output_details[0])

    if prediction.ndim == 2:
        probabilities = prediction[0].astype(np.float32)
    else:
        probabilities = prediction.reshape(-1).astype(np.float32)

    if probabilities.size == 0:
        raise ValueError("TFLite model returned empty output.")

    class_index = int(np.argmax(probabilities))

    if class_index >= len(classes):
        raise ValueError(
            f"Predicted class index {class_index} is outside classes list. Classes length: {len(classes)}."
        )

    predicted_class = str(classes[class_index])
    confidence = float(probabilities[class_index])

    return predicted_class, confidence, probabilities.tolist()
