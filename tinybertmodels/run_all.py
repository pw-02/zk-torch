import os
import subprocess
import time
import psutil

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

# Output log files
log_file = "run_log.txt"
stats_file = "process_stats.txt"

# Start timing and memory tracking
start_time = time.time()
process = psutil.Process(os.getpid())
max_memory = 0

with open(log_file, "w") as log:
    for file in os.listdir(yaml_dir):
        if file.endswith(".yaml"):
            yaml_path = os.path.join(yaml_dir, file)
            cmd = [
                "cargo", "run", "--release", "--bin", "zk_torch",
                "--features", "mock_prove,fold", "--", yaml_path
            ]

            print(f"🚀 Running: {' '.join(cmd)}")
            log.write(f"\n=== Running {file} ===\n")

            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Log stdout and stderr
            log.write(result.stdout)
            log.write(result.stderr)

            # Update max memory usage
            mem = process.memory_info().rss / (1024 ** 2)
            if mem > max_memory:
                max_memory = mem

            if result.returncode == 0:
                print(f"✅ {file} completed successfully\n")
            else:
                print(f"❌ {file} failed (check log)\n")

            # Cleanup step
            print("🧹 Cleaning up generated files...")
            for f in cleanup_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        print(f"  - Removed file: {f}")
                    except Exception as e:
                        print(f"  ⚠️ Could not remove {f}: {e}")
            print("Cleanup complete.\n")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Write summary stats
with open(stats_file, "w") as stats:
    stats.write(f"Total elapsed time: {elapsed_time:.2f} seconds\n")
    stats.write(f"Peak memory usage: {max_memory:.2f} MB\n")

print("🎯 All YAML files processed.")
print(f"🕒 Total elapsed time: {elapsed_time:.2f} seconds")
print(f"💾 Peak memory usage: {max_memory:.2f} MB")
print(f"📄 See {log_file} and {stats_file} for details.")
