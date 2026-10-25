#!/usr/bin/env python3
"""Run all 15 benchmark problems through the remote Emmy runner (runner.riftstack.ai)."""

import subprocess
import sys
import time

PROBLEMS = [
    ("Pointwise", "p00_scalar_add"),
    ("Pointwise", "p01_vector_add"),
    ("Pointwise", "p02_scale_shift"),
    ("Pointwise", "p03_relu_mul"),
    ("Reduction", "r01_array_sum"),
    ("Reduction", "r02_big_sum"),
    ("Reduction", "r03_mean"),
    ("Reduction", "r04_row_sum"),
    ("Matmul", "m01_matmul_512"),
    ("Matmul", "m02_matmul_rect"),
    ("Fused", "f01_rmsnorm"),
    ("Fused", "f02_row_softmax"),
    ("Fused", "f03_l2norm"),
    ("Attention", "a01_attention"),
    ("Attention", "a02_attention_128"),
]


def main():
    print("=" * 85)
    print("Running Emmy Hackathon 2026 Test Suite on Remote RTX 5090 (runner.riftstack.ai)")
    print("=" * 85)
    print(f"{'Category':<12} | {'Problem':<18} | {'Status':<6} | {'Runner Output'}")
    print("-" * 85)

    passed_count = 0
    start_all = time.time()

    for cat, prob in PROBLEMS:
        res = subprocess.run(
            [sys.executable, "submit.py", prob, "--compiler", "compile.py"],
            capture_output=True,
            text=True,
        )
        out = (res.stdout + "\n" + res.stderr).strip()
        status = "PASS" if "PASS" in out else "FAIL"
        if status == "PASS":
            passed_count += 1

        # Extract the line showing the register count and the pass summary
        lines = [l.strip() for l in out.splitlines() if "PASS" in l or "FAIL" in l or "ERROR" in l]
        summary = lines[-1] if lines else out.replace("\n", " ")
        print(f"{cat:<12} | {prob:<18} | {status:<6} | {summary}", flush=True)

    elapsed_all = time.time() - start_all
    print("=" * 85)
    print(f"Summary: {passed_count}/{len(PROBLEMS)} passed in {elapsed_all:.2f}s total on remote hardware")
    print("=" * 85)


if __name__ == "__main__":
    main()
