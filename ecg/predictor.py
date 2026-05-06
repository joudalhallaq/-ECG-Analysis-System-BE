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
_SIGNATURE_RUNNER = None
_INPUT_NAME = None
_OUTPUT_NAME = None
_CLASSES = None
_TRAIN_MEAN = None
_TRAIN_STD = None
_CONFIG = None


def load_assets():
    global _INTERPRETER, _SIGNATURE_RUNNER, _INPUT_NAME, _OUTPUT_NAME
    global _CLASSES, _TRAIN_MEAN, _TRAIN_STD, _CONFIG

    if _INTERPRETER is None:
        _INTERPRETER = Interpreter(model_path=str(MODEL_PATH))
        _INTERPRETER.allocate_tensors()

        signature_list = _INTERPRETER.get_signature_list()

        if signature_list:
            signature_key = "serving_default"

            if signature_key not in signature_list:
                signature_key = list(signature_list.keys())[0]

            signature_info = signature_list[signature_key]

            _INPUT_NAME = signature_info["inputs"][0]
            _OUTPUT_NAME = signature_info["outputs"][0]
            _SIGNATURE_RUNNER = _INTERPRETER.get_signature_runner(signature_key)
        else:
            _SIGNATURE_RUNNER = None

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

    return (
        _INTERPRETER,
        _SIGNATURE_RUNNER,
        _INPUT_NAME,
        _OUTPUT_NAME,
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
        raise ValueError(
            "No numeric values found in uploaded CSV file. Please upload an ECG signal CSV file."
        )

    return values


def get_expected_shape(interpreter, config):
    input_details = interpreter.get_input_details()

    expected_length = 1000
    expected_leads = 12

    if input_details:
        shape = list(input_details[0].get("shape", []))

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


def prepare_signal(file_path):
    (
        interpreter,
        _,
        _,
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

    expected_length, expected_leads = get_expected_shape(interpreter, config)

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
        signature_runner,
        input_name,
        output_name,
        classes,
        _,
        _,
        _,
    ) = load_assets()

    x = prepare_signal(file_path)

    if signature_runner is not None:
        result = signature_runner(**{input_name: x})

        if output_name in result:
            prediction = result[output_name]
        else:
            prediction = list(result.values())[0]
    else:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if not input_details:
            raise ValueError("TFLite input details are empty.")

        if not output_details:
            raise ValueError("TFLite output details are empty.")

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        expected_shape = list(input_details[0]["shape"])
        actual_shape = list(x.shape)

        if expected_shape != actual_shape:
            interpreter.resize_tensor_input(input_index, actual_shape)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_index = input_details[0]["index"]
            output_index = output_details[0]["index"]

        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)

    prediction = np.array(prediction, dtype=np.float32)

    if prediction.ndim == 2:
        probabilities = prediction[0]
    else:
        probabilities = prediction.reshape(-1)

    if probabilities.size == 0:
        raise ValueError("TFLite model returned empty output.")

    class_index = int(np.argmax(probabilities))

    if class_index >= len(classes):
        raise ValueError(
            f"Class index {class_index} is out of range. Classes length: {len(classes)}"
        )

    predicted_class = str(classes[class_index])
    confidence = float(probabilities[class_index])

    return predicted_class, confidence, probabilities.tolist()
