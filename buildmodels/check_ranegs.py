# buildmodels/check_ranges.py
import onnx
import numpy as np
import math
import sys

def check_ranges(model_path, scale_factor_log=8):
    scale = 2 ** scale_factor_log
    model = onnx.load(model_path)

    min_val, max_val = float("inf"), float("-inf")
    unscaled_max_abs = 0.0

    for init in model.graph.initializer:
        arr = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
        unscaled_max_abs = max(unscaled_max_abs, float(np.abs(arr).max()))
        qarr = arr * scale
        min_val = min(min_val, float(qarr.min()))
        max_val = max(max_val, float(qarr.max()))

    print(f"Unscaled Max Absolute Value: {unscaled_max_abs:.3f}")

    print(f"Scale factor log: {scale_factor_log} (scale={scale})")
    print(f"Scaled value range: {min_val:.3f} .. {max_val:.3f}")

    # recommend cq_range_log (enough to cover max abs value)
    max_abs = max(abs(min_val), abs(max_val))
    cq_range_log = math.ceil(math.log2(max_abs + 1)) if max_abs > 0 else 1

    # recommend cq_range_lower_log (smallest nonzero abs value)
    cq_range_lower_log = None
    if min_val != 0 or max_val != 0:
        nonzero_min = np.min(
            [abs(v) for init in model.graph.initializer
             for v in np.frombuffer(init.raw_data, dtype=np.float32).flatten() if v != 0]
        ) * scale
        cq_range_lower_log = math.floor(math.log2(nonzero_min)) if nonzero_min > 0 else 0

    print(f"Recommended cq_range_log: {cq_range_log}")
    if cq_range_lower_log is not None:
        print(f"Recommended cq_range_lower_log: {cq_range_lower_log}")
    else:
        print("All values are zero, lower bound can stay at 0.")

if __name__ == "__main__":
    model_path = "sample.onnx"
    scale_factor_log = 3

    check_ranges(model_path, scale_factor_log)
