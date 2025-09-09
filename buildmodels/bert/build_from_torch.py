import torch
from transformers import AutoModel, AutoTokenizer
import onnx 
from replace_reshape_trans import replace_reshape_transpose

# 1. Load model with standard attention
model_name = "google/bert_uncased_L-2_H-128_A-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, attn_implementation="eager")  # 👈 important
model.eval()

# 2. Create fixed dummy input
# dummy_input = torch.ones((1, 1), dtype=torch.int64)

dummy_input = torch.ones((1, 1), dtype=torch.int64)
dummy_input = tokenizer(
    ["hello"], 
    padding="max_length", 
    truncation=True, 
    max_length=1,   # 👈 exactly 1 token
    return_tensors="pt"
)
output_onnx = "buildmodels/bert/bert_fixed.onnx"
# 3. Export to ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    output_onnx,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],  # only include outputs you need
    dynamic_axes=None,   # 🚫 no symbolic dims
    opset_version=11     # now works, because no SDPA op
)


# Load the original model
model = onnx.load(output_onnx)

# Apply the pattern replacement
model = replace_reshape_transpose(model)

# Check the model for correctness
#onnx.checker.check_model(model)

# Save the modified model
onnx.save(model, output_onnx)





print("ONNX model saved as bert_fixed.onnx")
