import onnx

# Load model
model = onnx.load("buildmodels/bert/tinybert.onnx")

# Edit input shapes
for input_tensor in model.graph.input:
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_param:  # dynamic dimension like "batch_size"
            if dim.dim_param == "batch_size":
                dim.dim_value = 1
            elif dim.dim_param == "sequence":
                dim.dim_value = 384

# Save back
onnx.save(model, "buildmodels/bert/tinybert.onnx")
