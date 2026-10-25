# Emmy Kernel IR to PTX Compiler

A high-performance optimizing compiler that directly translates **Emmy Kernel IR** into native **NVIDIA PTX (ISA 8.5+ / sm_89 & sm_120)** assembly, completely bypassing the CUDA C++ / `nvcc` compilation stage.

Developed for the **Emmy Hackathon 2026**. Validated on real NVIDIA hardware (**RTX 5090 / sm_120**).

---

## 🚀 Benchmark Results (15/15 PASS on RTX 5090)

Evaluated via the remote live runner (`runner.riftstack.ai:48701`) executing on real hardware:

| Category | Problem | Kernel Name | Registers | Max Abs Error | Kernel Latency | Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Pointwise** | `p00_scalar_add` | `k_add_pointwise` | 16 | 0.000e+00 | **23.20 µs** | ✅ PASS |
| | `p01_vector_add` | `k_add_pointwise` | 16 | 0.000e+00 | **43.07 µs** | ✅ PASS |
| | `p02_scale_shift` | `k_add_pointwise` | 16 | 0.000e+00 | **16.03 µs** | ✅ PASS |
| | `p03_relu_mul` | `k_mul_pointwise` | 16 | 0.000e+00 | **16.38 µs** | ✅ PASS |
| **Reduction** | `r01_array_sum` | `k_sum_1_reduce` | 24 | 3.815e-06 | **23.36 µs** | ✅ PASS |
| | `r02_big_sum` | `k_sum_1_reduce` | 24 | 1.526e-05 | **44.06 µs** | ✅ PASS |
| | `r03_mean` | `k_mean_ffcca3` | 16 | 9.313e-10 | **31.94 µs** | ✅ PASS |
| | `r04_row_sum` | `k_sum_1_reduce` | 24 | 5.722e-06 | **23.87 µs** | ✅ PASS |
| **Matmul** | `m01_matmul_512` | `k_matmul_2967be` | 30 | 8.392e-05 | **50.27 µs** | ✅ PASS |
| | `m02_matmul_rect` | `k_matmul` | 28 | 1.526e-05 | **15.04 µs** | ✅ PASS |
| **Fused** | `f01_rmsnorm` | `k_mean_cbd1de` | 18 | 1.907e-06 | **16.38 µs** | ✅ PASS |
| | `f02_row_softmax` | `k_softmax_05263d` | 20 | 7.451e-09 | **15.94 µs** | ✅ PASS |
| | `f03_l2norm` | `k_div_reduce` | 18 | 2.980e-08 | **16.93 µs** | ✅ PASS |
| **Attention** | `a01_attention` | `k_matmul_softmax_reduce_b55365` | 56 | 6.557e-07 | **212.61 µs** | ✅ PASS |
| | `a02_attention_128` | `k_matmul_softmax_reduce_6f0029` | 48 | 4.470e-07 | **97.18 µs** | ✅ PASS |

**Score: 15 / 15 (100% Correctness across all 5 tiers in 43.69s total on RTX 5090)**

---

## 🏗️ Compiler Architecture

```
                 JSON Deserialization
Kernel IR (AST) ───────────────────────► Emmy Graph Representation
                                                   │
                                                   ▼
                                         [AST Optimization Passes]
                                         ├─ Pass 1: Algebraic Simplification
                                         ├─ Pass 2: Address Strength Reduction
                                         ├─ Pass 3: Immediate-Operand Folding
                                         └─ Pass 4: FMA Fusion Pattern Matcher
                                                   │
                                                   ▼
                                         [Type & Layout Analysis]
                                         ├─ Shared Memory Pool Layout
                                         ├─ Warp-Level Sync & Divergence
                                         └─ Vector Width (v1, v2, v4)
                                                   │
                                                   ▼
                                            [PTX Emission]
                                         ├─ Asynchronous Copy (cp.async)
                                         ├─ Hardware Warp Shuffles (shfl.sync)
                                         ├─ Multi-dtype (f32, f16, bf16)
                                         └─ Compact Register Allocation
                                                   │
                                                   ▼
                                            Target PTX Output
```

### Key Lowering Components

1. **Direct AST Tree Walking & Environment Tracking**
   - Implements a visitor-based statement lowerer in `compile_stmts()` handling:
     `Tile`, `Loop`, `StridedLoop`, `Load`, `Write`, `Assign`, `Accum`, `Let`, `Init`, `Alloc`, `Sync`, `WarpReduce`, `CpAsyncCopy`, `CpAsyncCommit`, `CpAsyncWait`.
   - Maintains an SSA-style register environment (`PTXEmitter.env`) with dedicated register pools:
     - `.reg .pred %p<N>` (predicates)
     - `.reg .b32  %r<N>` (32-bit integer & bitwise)
     - `.reg .b64  %rd<N>` (64-bit pointers)
     - `.reg .f32  %f<N>` (single-precision floats)
     - `.reg .b16  %h<N>` (half-precision f16 / bf16)

2. **Hardware Asynchronous Tile Loading (`cp.async`)**
   - Software-pipelined asynchronous DMA loads directly bypassing the register file:
     ```ptx
     cp.async.ca.shared.global [%r_dst], [%rd_src], 16;
     cp.async.commit_group;
     cp.async.wait_group 2;
     ```
   - Allows memory transfers to overlap directly with tensor core arithmetic.

3. **Optimized Address Calculation (Strength Reduction)**
   - Fuses multi-dimensional base + stride pointer calculations into 64-bit integer multiply-accumulate instructions:
     ```ptx
     mad.wide.s32 %rd_addr, %r_idx, 4, %rd_base;
     ```
   - Completely removes intermediate register copies and arithmetic chains.

4. **Warp-Level Cooperative Reductions (`shfl.sync`)**
   - Translates high-level tree reductions (`WarpReduce`) into hardware warp-shuffle butterfly steps:
     ```ptx
     shfl.sync.down.b32 %f_tmp, %f_val, 16, 0x1f, 0xffffffff;
     add.f32 %f_val, %f_val, %f_tmp;
     ```
   - Automatically handles arbitrary reduction operations (`add`, `maximum`) without shared memory bank conflicts.

5. **Tensor Core Matrix Multiply-Accumulate (`mma.sync` & `ldmatrix`)**
   - Native support for Tensor Core instructions (`m16n8k16` / `m8n8`):
     ```ptx
     ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r_smem];
     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1, %f2, %f3, %f4}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%f1, %f2, %f3, %f4};
     ```
   - Automatically allocates per-thread fragment register pools (`RegFragment`), handles matrix transpose flags (`.trans`), chunked accumulation promotion (`FragmentPromote`), and fragment output downconversion (`RegStore`).

6. **Multi-Precision Support (`f32`, `f16`, `bf16`)**
   - Dynamic instruction selection based on node dtypes:
     - Native `.f32` instructions for single-precision workloads.
     - Native `.b16` register handling and conversions (`cvt.rn.f16.f32`, `cvt.f32.f16`) for mixed-precision operations.
     - Accurate constant encoding into IEEE-754 hexadecimal literals (`float_to_ptx_hex`).

---

## 🧪 Automated Test Suite (21 Tests, 100% Pass)

A complete offline unit test suite is included in `tests/test_compiler.py` covering all 15 problem IRs, Tensor Core lowering, async copies, AST optimization passes, and precision types without needing GPU access:

```bash
# Run the complete test suite
python -m unittest discover tests -v
```

Output:
```text
Ran 21 tests in 0.097s — OK (21/21 passing)
```

---

## 💻 Requirements & Usage

### Prerequisites
- Python 3.10+
- `emmy-ml==0.3.11`
- `numpy`, `requests`

```bash
pip install emmy-ml==0.3.11 numpy requests
```

### Compile an IR to PTX
The compiler strictly implements the required CLI contract:
```bash
python compile.py problems/m01_matmul_512/k_matmul_2967be.ir.json > out.ptx
```

### Validate against the Live RTX 5090 Runner
```bash
# Single problem test
python submit.py m01_matmul_512 --compiler compile.py

# Full test suite execution on live hardware
python -c "
import subprocess, sys
problems = [
    'p00_scalar_add', 'p01_vector_add', 'p02_scale_shift', 'p03_relu_mul',
    'r01_array_sum', 'r02_big_sum', 'r03_mean', 'r04_row_sum',
    'm01_matmul_512', 'm02_matmul_rect',
    'f01_rmsnorm', 'f02_row_softmax', 'f03_l2norm',
    'a01_attention', 'a02_attention_128'
]
for p in problems:
    res = subprocess.run([sys.executable, 'submit.py', p, '--compiler', 'compile.py'], capture_output=True, text=True)
    status = 'PASS' if 'PASS' in (res.stdout + res.stderr) else 'FAIL'
    print(f'{p:<20}: {status}')
"
```

---

## 📁 Repository Structure

```
├── compile.py          # The core PTX optimizing compiler (IR -> PTX)
├── submit.py           # Evaluation runner client (targets runner.riftstack.ai)
├── tests/              # Automated unit and integration test suite
│   └── test_compiler.py # 21 comprehensive offline tests
├── problems/           # 15 benchmark problem definitions (IR json, ref, meta)
│   ├── p00_scalar_add/
│   ├── ...
│   └── a02_attention_128/
├── docs/               # Specifications and Runner API details
├── MVP.md              # Minimum viable product specifications
└── README.md           # Compiler architecture and benchmark documentation
```

---

## 👥 Authors
Emmy Hackathon 2026 Submission Team

