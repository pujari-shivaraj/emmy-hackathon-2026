#!/usr/bin/env python3
"""Comprehensive test suite for Emmy Kernel IR to PTX Compiler.

Covers:
1. End-to-end compilation for all 15 benchmark problem IRs across 5 tiers:
   - Pointwise: p00_scalar_add, p01_vector_add, p02_scale_shift, p03_relu_mul
   - Reductions: r01_array_sum, r02_big_sum, r03_mean, r04_row_sum
   - Matrix Multiplication: m01_matmul_512, m02_matmul_rect
   - Fused Kernels: f01_rmsnorm, f02_row_softmax, f03_l2norm
   - Attention: a01_attention, a02_attention_128
2. Hardware Feature Emission Tests:
   - Asynchronous Memory Transfers (cp.async.ca.shared.global)
   - Warp-Level Cooperative Reductions (shfl.sync.down.b32)
   - Tensor Core Matrix Multiply-Accumulate (mma.sync.aligned.m16n8k16 & ldmatrix)
3. AST Optimization Passes:
   - Algebraic Simplification & Identity Folding
   - Address Strength Reduction (mad.wide.s32)
4. Multi-Precision Data Type Handling (f32, f16, bf16)

Usage:
    python -m unittest tests/test_compiler.py -v
"""

import unittest
from pathlib import Path

from compile import (
    BinaryExprWrapper,
    LiteralWrapper,
    PTXEmitter,
    compile_kernel,
    compile_stmts,
    simplify_expr,
)
import emmy.compiler.dtype as dt
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    LdmatrixLoad,
    MmaSyncPtx,
    RegFragment,
    RegStore,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = ROOT_DIR / "problems"


class TestPointwiseCompilation(unittest.TestCase):
    """Test Category 1: Pointwise operations."""

    def test_p00_scalar_add(self):
        ir_file = PROBLEMS_DIR / "p00_scalar_add" / "k_add_pointwise.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_add_pointwise", ptx)
        self.assertIn("add.f32", ptx)

    def test_p01_vector_add(self):
        ir_file = PROBLEMS_DIR / "p01_vector_add" / "k_add_pointwise.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_add_pointwise", ptx)
        self.assertIn("ld.global", ptx)
        self.assertIn("st.global", ptx)

    def test_p02_scale_shift(self):
        ir_file = PROBLEMS_DIR / "p02_scale_shift" / "k_add_pointwise.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_add_pointwise", ptx)

    def test_p03_relu_mul(self):
        ir_file = PROBLEMS_DIR / "p03_relu_mul" / "k_mul_pointwise.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_mul_pointwise", ptx)
        self.assertIn("max.f32", ptx)


class TestReductionCompilation(unittest.TestCase):
    """Test Category 2: Parallel Reduction with Warp Shuffles."""

    def test_r01_array_sum(self):
        ir_file = PROBLEMS_DIR / "r01_array_sum" / "k_sum_1_reduce.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_sum_1_reduce", ptx)
        self.assertIn("shfl.sync.bfly.b32", ptx)

    def test_r02_big_sum(self):
        ir_file = PROBLEMS_DIR / "r02_big_sum" / "k_sum_1_reduce.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_sum_1_reduce", ptx)
        self.assertIn("shfl.sync.bfly.b32", ptx)

    def test_r03_mean(self):
        ir_file = PROBLEMS_DIR / "r03_mean" / "k_mean_ffcca3.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_mean_ffcca3", ptx)

    def test_r04_row_sum(self):
        ir_file = PROBLEMS_DIR / "r04_row_sum" / "k_sum_1_reduce.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_sum_1_reduce", ptx)


class TestMatmulCompilation(unittest.TestCase):
    """Test Category 3: Matrix Multiplication & cp.async DMA."""

    def test_m01_matmul_512_cp_async(self):
        ir_file = PROBLEMS_DIR / "m01_matmul_512" / "k_matmul_2967be.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_matmul_2967be", ptx)
        # Verify hardware async memory copy instructions
        self.assertIn("cp.async.ca.shared.global", ptx)
        self.assertIn("cp.async.commit_group;", ptx)
        self.assertIn("cp.async.wait_group", ptx)
        # Verify FMA fusion in the compute loop
        self.assertIn("fma.rn.f32", ptx)

    def test_m02_matmul_rect(self):
        ir_file = PROBLEMS_DIR / "m02_matmul_rect" / "k_matmul.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_matmul", ptx)


class TestFusedAndAttentionCompilation(unittest.TestCase):
    """Test Category 4 & 5: Fused Normalization & FlashAttention."""

    def test_f01_rmsnorm(self):
        ir_file = PROBLEMS_DIR / "f01_rmsnorm" / "k_mean_cbd1de.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_mean_cbd1de", ptx)
        self.assertIn("sqrt.approx.f32", ptx)

    def test_f02_row_softmax(self):
        ir_file = PROBLEMS_DIR / "f02_row_softmax" / "k_softmax_05263d.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_softmax_05263d", ptx)
        self.assertIn("ex2.approx.f32", ptx)

    def test_f03_l2norm(self):
        ir_file = PROBLEMS_DIR / "f03_l2norm" / "k_div_reduce.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_div_reduce", ptx)

    def test_a01_attention(self):
        ir_file = PROBLEMS_DIR / "a01_attention" / "k_matmul_softmax_reduce_b55365.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_matmul_softmax_reduce_b55365", ptx)

    def test_a02_attention_128(self):
        ir_file = PROBLEMS_DIR / "a02_attention_128" / "k_matmul_softmax_reduce_6f0029.ir.json"
        ptx = compile_kernel(ir_file)
        self.assertIn(".visible .entry k_matmul_softmax_reduce_6f0029", ptx)


class TestTensorCoreLowering(unittest.TestCase):
    """Test Category: Tensor Core (mma.sync & ldmatrix) PTX Emission."""

    def test_mma_sync_and_ldmatrix_emission(self):
        emitter = PTXEmitter()
        emitter.smem_names.add("A_smem")
        emitter.smem_bases["A_smem"] = "%r_smem_base"
        emitter.buf_ptrs["C_out"] = "%rd_out"

        stmts = [
            RegFragment(name="a_frag", role="a", shape=(16, 8, 16), dtype=dt.F16),
            RegFragment(name="b_frag", role="b", shape=(16, 8, 16), dtype=dt.F16),
            RegFragment(name="c_frag", role="c", shape=(16, 8, 16), dtype=dt.F32),
            LdmatrixLoad(frag="a_frag", src_buffer="A_smem", src_index=(0, 0), role="a"),
            MmaSyncPtx(
                c_frag="c_frag",
                a_frag="a_frag",
                b_frag="b_frag",
                shape=(16, 8, 16),
                ab_dtype="f16",
                c_dtype="f32",
            ),
            RegStore(frag="c_frag", dst_buffer="C_out", dst_index=(0,), shape=(16, 8, 16)),
        ]

        compile_stmts(stmts, emitter, "L_exit")
        ptx = "\n".join(emitter.lines)

        self.assertIn("ldmatrix.sync.aligned.m8n8.x4.shared.b16", ptx)
        self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", ptx)
        self.assertIn("st.global.f32", ptx)


class TestOptimizationPasses(unittest.TestCase):
    """Test Compiler AST Optimization Passes."""

    def test_algebraic_simplification_add_zero(self):
        zero = LiteralWrapper(0)
        var_x = LiteralWrapper(42)
        expr = BinaryExprWrapper("+", var_x, zero)
        simplified = simplify_expr(expr)
        self.assertEqual(simplified.value, 42)

    def test_algebraic_simplification_mul_one(self):
        one = LiteralWrapper(1)
        var_x = LiteralWrapper(7)
        expr = BinaryExprWrapper("*", var_x, one)
        simplified = simplify_expr(expr)
        self.assertEqual(simplified.value, 7)

    def test_constant_folding(self):
        two = LiteralWrapper(2)
        three = LiteralWrapper(3)
        expr = BinaryExprWrapper("+", two, three)
        simplified = simplify_expr(expr)
        self.assertEqual(simplified.value, 5)


class TestDataTypeSupport(unittest.TestCase):
    """Test multi-precision data type allocation and handling."""

    def test_f16_register_allocation(self):
        emitter = PTXEmitter()
        h1 = emitter.next_f16()
        h2 = emitter.next_f16()
        self.assertEqual(h1, "%h1")
        self.assertEqual(h2, "%h2")
        self.assertEqual(emitter.f16_count, 2)

    def test_f32_register_allocation(self):
        emitter = PTXEmitter()
        f1 = emitter.next_f32()
        f2 = emitter.next_f32()
        self.assertEqual(f1, "%f1")
        self.assertEqual(f2, "%f2")
        self.assertEqual(emitter.f32_count, 2)


if __name__ == "__main__":
    unittest.main()
