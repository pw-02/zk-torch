import os
import subprocess

# Path to your YAML files
yaml_dir = "tinybertmodels/splits"

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

            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Log both stdout and stderr
            log.write(result.stdout)
            log.write(result.stderr)

            # Optional: print progress to terminal
            if result.returncode == 0:
                print(f"✅ {file} completed successfully\n")
            else:
                print(f"❌ {file} failed (check log)\n")

print("🎯 All YAML files processed. See run_log.txt for details.")
