from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load model and tokenizer directly from Hugging Face
# model_name = "bert-mini-uncased-whole-word-masking-finetuned-squad"
model_name = "prajjwal1/bert-mini"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Save as PyTorch checkpoint
torch.save(model.state_dict(), "model.pytorch")

# Export to ONNX
model.eval()
dummy_input = torch.ones((1, 384), dtype=torch.int64)
torch.onnx.export(
    model,
    (dummy_input, dummy_input, dummy_input),
    "bert.onnx",
    input_names=["input_ids", "input_mask", "segment_ids"],
    output_names=["output_start_logits", "output_end_logits"],
    opset_version=14
)
