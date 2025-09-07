import onnx
import onnxruntime as ort
import numpy as np
import json

def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)
    
def format_model_input(input_data_path, expected_shape, input_type, idx=0):
    """Format input tensor from JSON to match ONNX model expectations."""
    expected_shape = [-1 if dim == 'batch_size' else dim for dim in expected_shape]
    input_data = load_json_input(input_data_path)['input_data']
    if 'yolo' in str(input_data_path).lower():
        expected_shape = [1, 3, 416, 416]  # Example for YOLO models
    if input_type == 'tensor(float)':
        reshaped_input = np.array(input_data, dtype=np.float32).reshape(expected_shape)
    elif input_type == 'tensor(int64)':
        reshaped_input = np.array(input_data[idx] if expected_shape else input_data[0][0], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    if 'gpt' in str(input_data_path).lower():
        reshaped_input = np.reshape(input_data, (1, 64))

    return reshaped_input

# ---- Load model + input ----
onnx_model = "tiny_mnist_fixed.onnx"
input_json = "tiny_mnist_input.json"

with open(input_json) as f:
    data = json.load(f)


# ---- Load ONNX model to inspect graph ----
onnx_model_obj = onnx.load(onnx_model)
onnx_outputs = [node.output[0] for node in onnx_model_obj.graph.node]

# # ---- Create inference session ----
# session = ort.InferenceSession(onnx_model)
# input_name = session.get_inputs()[0].name
# input_shape = session.get_inputs()[0].shape
# input_type = session.get_inputs()[0].type
# input_tensor = format_model_input(input_json, input_shape, input_type)
# outputs = session.run(None, {input_name: input_tensor})


model = onnx.load(onnx_model)
model.graph.ClearField('output')
shape_info = onnx.shape_inference.infer_shapes(model)

for node in shape_info.graph.node:
    for output_name in node.output:
        if not any(o.name == output_name for o in model.graph.output):
            output_info = onnx.ValueInfoProto()
            output_info.name = output_name
            model.graph.output.append(output_info)

session = ort.InferenceSession(model.SerializeToString())
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type
print(f"Input name: {input_name}, shape: {input_shape}, type: {input_type}")
input_data = format_model_input(input_json, input_shape, input_type)

outputs = session.run(None, {input_name: input_data})


# ---- Scan through results ----
max_val = 0.0
for name, arr in zip(onnx_outputs, outputs):
    arr = np.array(arr)
    node_max = np.max(np.abs(arr))
    node_min = np.min(arr)
    print(f"{name:30s} min={node_min:.4f}, max={np.max(arr):.4f}")
    max_val = max(max_val, node_max)

print("\n==== Summary ====")
print("Maximum absolute activation value:", max_val)
