import os
import subprocess

# Path to your YAML files
yaml_dir = "tinybertmodels/splits"

# Files to remove after each run
cleanup_files = [
    "layer_setup",
    "models",
    "setups",
    "final_proofs",
    "modelsEnc",
    "outputsEnc",
    "acc_proofs"
]

# Optional: log output to a file
log_file = "run_log.txt"

with open(log_file, "w") as log:
    for file in os.listdir(yaml_dir):
        if file.endswith(".yaml"):
            yaml_path = os.path.join(yaml_dir, file)
            cmd = [
                "cargo", "run", "--release", "--bin", "zk_torch",
                "--features", "fold", "--", yaml_path
            ]

            print(f"🚀 Running: {' '.join(cmd)}")
            log.write(f"\n=== Running {file} ===\n")

            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Log both stdout and stderr
            log.write(result.stdout)
            log.write(result.stderr)

            if result.returncode == 0:
                print(f"✅ {file} completed successfully\n")
            else:
                print(f"❌ {file} failed (check log)\n")

            # --- Cleanup step ---
            print("🧹 Cleaning up generated files...")
            for f in cleanup_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        print(f"  - Removed file: {f}")
                    except Exception as e:
                        print(f"  ⚠️ Could not remove {f}: {e}")
            print("Cleanup complete.\n")

print("🎯 All YAML files processed. See run_log.txt for details.")
