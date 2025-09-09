import onnx

# Load model
path = "buildmodels/resnet50/resnet50_v1.onnx"
model = onnx.load(path)

# Edit input shapes
for inp in model.graph.input:
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_param:  # symbolic name found
            d.dim_param = ""   # clear it
            d.dim_value = 1    # set fixed size (e.g. 1)

# Save back
onnx.save(model, "buildmodels/resnet50/resnet50_v1.onnx")
