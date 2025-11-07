import os
import subprocess
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==== CONFIGURATION ====
yaml_dir = "tinybertmodels/splits"
num_parallel = 10  # number of processes to run concurrently

cleanup_files = [
    "layer_setup",
    "models",
    "setups",
    "final_proofs",
    "modelsEnc",
    "outputsEnc",
    "acc_proofs"
]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# ==== FUNCTIONS ====
def run_model(yaml_file):
    """Run one model and return timing + success info."""
    start_time = time.time()
    yaml_path = os.path.join(yaml_dir, yaml_file)
    cmd = [
        "cargo", "run", "--release", "--bin", "zk_torch",
        "--features", "mock_prove,fold", "--", yaml_path
    ]
    log_path = os.path.join(log_dir, f"{os.path.splitext(yaml_file)[0]}.log")

    with open(log_path, "w") as log:
        log.write(f"=== Running {yaml_file} ===\n")
        log.write(" ".join(cmd) + "\n\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            log.write(result.stdout)
            log.write(result.stderr)
        except Exception as e:
            log.write(f"Error running {yaml_file}: {e}\n")
            return yaml_file, False, 0

    # Cleanup after run
    for f in cleanup_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    elapsed = time.time() - start_time
    success = result.returncode == 0
    return yaml_file, success, elapsed


# ==== MAIN ====
if __name__ == "__main__":
    start_time = time.time()
    process = psutil.Process(os.getpid())
    max_memory = 0

    yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith(".yaml")]

    print(f"🚀 Running {len(yaml_files)} models with {num_parallel} parallel processes...\n")

    results = []
    with ProcessPoolExecutor(max_workers=num_parallel) as executor:
        futures = {executor.submit(run_model, f): f for f in yaml_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                yaml_file, success, elapsed = future.result()
                results.append((yaml_file, success, elapsed))
                mem = process.memory_info().rss / (1024 ** 2)
                if mem > max_memory:
                    max_memory = mem
                status = "✅" if success else "❌"
                print(f"{status} {yaml_file} finished in {elapsed:.1f}s")
            except Exception as e:
                print(f"❌ {file} failed: {e}")

    total_elapsed = time.time() - start_time

    # Write summary log
    summary_path = os.path.join(log_dir, "summary.txt")
    with open(summary_path, "w") as summary:
        summary.write(f"Total elapsed time: {total_elapsed:.2f} seconds\n")
        summary.write(f"Peak memory usage: {max_memory:.2f} MB\n")
        summary.write(f"Parallel processes: {num_parallel}\n\n")
        summary.write("Per-model results:\n")
        for yaml_file, success, elapsed in results:
            status = "OK" if success else "FAILED"
            summary.write(f"{yaml_file:30s} {status:8s} {elapsed:.2f}s\n")

    print("\n🎯 All YAML files processed.")
    print(f"🕒 Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"💾 Peak memory usage: {max_memory:.2f} MB")
    print(f"📄 Logs written to: {log_dir}/")
