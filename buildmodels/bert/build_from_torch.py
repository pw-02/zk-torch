import torch
from transformers import AutoModel, AutoTokenizer

# 1. Load model with standard attention
model_name = "google/bert_uncased_L-2_H-128_A-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, attn_implementation="eager")  # 👈 important
model.eval()

# 2. Create fixed dummy input
batch_size = 1
sequence_length = 16
dummy_input = tokenizer(
    ["Hello world!"],
    padding="max_length",
    truncation=True,
    max_length=sequence_length,
    return_tensors="pt"
)

# 3. Export to ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "bert_fixed.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],  # pooler_output only if present
    dynamic_axes=None,   # 🚫 no symbolic dims
    opset_version=11     # now works, because no SDPA op
)

print("ONNX model saved as bert_fixed.onnx")
