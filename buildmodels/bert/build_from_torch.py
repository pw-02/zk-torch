import torch
from transformers import AutoModel, AutoTokenizer

# 1. Load model and tokenizer from Hugging Face
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 2. Create a dummy input with fixed batch size and sequence length
batch_size = 1
sequence_length = 16  # choose a fixed length (e.g., 16, 32, 128 depending on your use case)
dummy_input = tokenizer(
    ["Hello world!"], 
    padding="max_length", 
    truncation=True, 
    max_length=sequence_length, 
    return_tensors="pt"
)

# 3. Export to ONNX (no dynamic axes)
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "bert_tiny_fixed.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes=None,   # 🚫 disables symbolic dims
    opset_version=14     # recommended opset for transformers
)

print("ONNX model saved as bert_tiny_fixed.onnx")
