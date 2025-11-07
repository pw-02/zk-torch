import os
import yaml

# Path where your model ONNX files are located
models_dir = "tinybertmodels/splits"

# Output directory for the generated YAML files
output_dir = "tinybertmodels/splits"
os.makedirs(output_dir, exist_ok=True)

# Loop through all ONNX model files in the directory
for file in os.listdir(models_dir):
    if file.endswith(".onnx"):
        model_name = os.path.splitext(file)[0]

        # Construct the YAML content
        yaml_content = {
            "task": model_name,
            "onnx": {
                "model_path": f"tinybertmodels/splits/{file}",
                # model_path: bert_fixed.onnx --- IGNORE ---
                "input_path": ""
            },
            "ptau": {
                "ptau_path": "challenge_0003",
                "pow_len_log": 22,
                "loaded_pow_len_log": 22
            },
            "sf": {
                "scale_factor_log": 3,
                "cq_range_log": 20,
                "cq_range_lower_log": 19
            },
            "prover": {
                "model_path": f"run_log.txt",
                "setup_path": "setups",
                "enc_model_path": "modelsEnc",
                "enc_input_path": "inputsEnc",
                "enc_output_path": "outputsEnc",
                "proof_path": "proofs",
                "acc_proof_path": "acc_proofs",
                "final_proof_path": "final_proofs",
                "enable_layer_setup": True
            },
            "verifier": {
                "enc_model_path": "modelsEnc",
                "enc_input_path": "inputsEnc",
                "enc_output_path": "outputsEnc",
                "proof_path": "proofs"
            }
        }

        # Write to YAML file
        output_path = os.path.join(output_dir, f"{model_name}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"✅ Generated: {output_path}")

print("\nAll YAML files generated successfully!")
