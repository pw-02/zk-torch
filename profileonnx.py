import onnxruntime as ort
import numpy as np
import json

# ---- Load model + input ----
onnx_model = "tiny_mnist_fixed.onnx"
input_json = "tiny_mnist_input.json"

with open(input_json) as f:
    data = json.load(f)

# Usually the ONNX input name matches the key in JSON
input_name = list(data.keys())[0]
x = np.array(data[input_name], dtype=np.float32)

# ---- Run ONNX with debug node outputs ----
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

session = ort.InferenceSession(onnx_model, sess_options, providers=["CPUExecutionProvider"])

# Get all node names
all_nodes = [n.name for n in session.get_outputs()]

# Trick: run model once to get final output
session.run(None, {input_name: x})

# ---- Dump intermediate outputs ----
# If you install: pip install onnxruntime-debug-node-output
# Then you can do:
from onnxruntime.capi._pybind_state import get_all_node_outputs

node_outputs = get_all_node_outputs(session, {input_name: x})

max_val = 0.0
for k, v in node_outputs.items():
    arr = np.array(v)
    node_max = np.max(np.abs(arr))
    print(f"{k:30s} min={np.min(arr):.4f}, max={np.max(arr):.4f}")
    max_val = max(max_val, node_max)

print("\n==== Summary ====")
print("Maximum absolute activation value:", max_val)
