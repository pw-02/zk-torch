import onnxruntime as ort
import numpy as np
import onnx

# Load model
onnx_model = "profile_mnist_fixed.onnx"
session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

# Load input (your profile_mnist_input.json → convert to numpy)
import json
with open("profile_mnist_input.json") as f:
    input_data = json.load(f)
input_name = session.get_inputs()[0].name
x = np.array(input_data[input_name], dtype=np.float32)

# Run model and grab all intermediate activations
from onnxruntime.extensions import get_library_path
from onnxruntime import InferenceSession

# Enable intermediate outputs
options = ort.SessionOptions()
options.enable_profiling = True

session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

# Run model
outputs = session.run(None, {input_name: x})

# Hook into model graph to inspect intermediates
# Simple way: re-export ONNX to numpy ops with "onnxruntime.training" or use `onnxruntime-debug-node-output`
