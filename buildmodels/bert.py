import os
import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def save_to_onnx(model, output_path):
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model.eval()

    dummy_input = torch.ones((1, 384), dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy_input, dummy_input, dummy_input),
        output_path,
        verbose=True,
        input_names=["input_ids", "input_mask", "segment_ids"],
        output_names=["output_start_logits", "output_end_logits"],
        opset_version=14,
        do_constant_folding=True,

    )

def main():
    output_dir = "buildmodels/bert"
    os.makedirs(output_dir, exist_ok=True)

    # Load directly from Hugging Face instead of TF checkpoint
    model = BertForQuestionAnswering.from_pretrained(
        "prajjwal1/bert-tiny"
    )

    # Save config.json for reference (like TF one)
    config_path = os.path.join(output_dir, "bert_config.json")
    with open(config_path, "w") as f:
        f.write(model.config.to_json_string())

    # Save as PyTorch state_dict
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pytorch"))

    # Save as ONNX
    save_to_onnx(model, os.path.join(output_dir, "tiny_bert.onnx"))

if __name__ == "__main__":
    main()
