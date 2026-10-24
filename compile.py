#!/usr/bin/env python3
"""Emmy Kernel IR to PTX compiler with Optimization Passes.

Optimization Pipeline:
1. Pass 1: Algebraic Simplification & Constant Folding
2. Pass 2: Address Strength Reduction (mad.wide.s32 & mad.lo.s32)
3. Pass 3: Immediate-Operand Folding (eliminates redundant movs)
4. Pass 4: FMA (Fused Multiply-Add) Pattern Matching

Usage:
    python compile.py path/to/<kernel>.ir.json > out.ptx
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph


def to_int(v) -> int:
    return int(v.value if hasattr(v, "value") else v)


def float_to_ptx_hex(val: float) -> str:
    packed = struct.pack(">f", float(val))
    return f"0f{packed.hex()}"


class BinaryExprWrapper:
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class LiteralWrapper:
    def __init__(self, value, dtype="int"):
        self.value = value
        self.dtype = dtype


def is_lit(expr, val=None) -> bool:
    if type(expr).__name__ in ("Literal", "LiteralWrapper"):
        if val is None:
            return True
        return expr.value == val
    return False


def get_lit(expr):
    return expr.value


# ==============================================================================
# PASS 1: Algebraic Simplification & Constant Folding
# ==============================================================================
def simplify_expr(expr):
    etype = type(expr).__name__
    if etype in ("Var", "Literal", "LiteralWrapper"):
        return expr

    if etype in ("BinaryExpr", "BinaryExprWrapper"):
        left = simplify_expr(expr.left)
        right = simplify_expr(expr.right)
        op = expr.op

        # Compile-time evaluation of constant literals
        if is_lit(left) and is_lit(right):
            lv = get_lit(left)
            rv = get_lit(right)
            if op == "+":
                return LiteralWrapper(lv + rv, "int")
            elif op == "-":
                return LiteralWrapper(lv - rv, "int")
            elif op == "*":
                return LiteralWrapper(lv * rv, "int")
            elif op == "/" and rv != 0:
                return LiteralWrapper(lv // rv, "int")
            elif op == "%" and rv != 0:
                return LiteralWrapper(lv % rv, "int")
            elif op == "<":
                return LiteralWrapper(1 if lv < rv else 0, "int")
            elif op == "<=":
                return LiteralWrapper(1 if lv <= rv else 0, "int")
            elif op == ">":
                return LiteralWrapper(1 if lv > rv else 0, "int")
            elif op == ">=":
                return LiteralWrapper(1 if lv >= rv else 0, "int")
            elif op == "==":
                return LiteralWrapper(1 if lv == rv else 0, "int")
            elif op == "!=":
                return LiteralWrapper(1 if lv != rv else 0, "int")

        # Identity rules
        if op == "+":
            if is_lit(right, 0):
                return left
            if is_lit(left, 0):
                return right
        elif op == "-":
            if is_lit(right, 0):
                return left
        elif op == "*":
            if is_lit(right, 1):
                return left
            if is_lit(left, 1):
                return right
            if is_lit(right, 0) or is_lit(left, 0):
                return LiteralWrapper(0, "int")
        elif op == "/":
            if is_lit(right, 1):
                return left

        return BinaryExprWrapper(op, left, right)

    if etype == "TernaryExpr":
        cond = simplify_expr(expr.cond)
        if is_lit(cond):
            return simplify_expr(expr.if_true) if get_lit(cond) != 0 else simplify_expr(expr.if_false)
        return type(expr)(cond, simplify_expr(expr.if_true), simplify_expr(expr.if_false))

    return expr


class PTXEmitter:
    def __init__(self):
        self.pred_count = 0
        self.f32_count = 0
        self.b32_count = 0
        self.b64_count = 0
        self.label_count = 0
        self.lines: list[str] = []
        # SSA name -> register name
        self.env: dict[str, str] = {}
        # Buffer name -> 64-bit base register name
        self.buf_ptrs: dict[str, str] = {}
        # Shape info: buffer name -> list of int
        self.buf_shapes: dict[str, list[int]] = {}
        # Set of shared memory buffer names
        self.smem_names: set[str] = set()
        # Mapping from smem buffer name to register holding base address
        self.smem_bases: dict[str, str] = {}
        # Names initialized via Init (carried state)
        self.init_names: set[str] = set()
        # FMA tracking: mul_name -> (arg0_reg, arg1_reg)
        self.mul_defs: dict[str, tuple[str, str]] = {}

    def next_pred(self) -> str:
        self.pred_count += 1
        return f"%p{self.pred_count}"

    def next_f32(self) -> str:
        self.f32_count += 1
        return f"%f{self.f32_count}"

    def next_b32(self) -> str:
        self.b32_count += 1
        return f"%r{self.b32_count}"

    def next_b64(self) -> str:
        self.b64_count += 1
        return f"%rd{self.b64_count}"

    def next_label(self, prefix: str = "L") -> str:
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    def emit(self, line: str):
        self.lines.append(line)

    # ==========================================================================
    # PASS 3: Immediate-Operand Folding
    # ==========================================================================
    def compile_expr(self, expr) -> str:
        expr = simplify_expr(expr)
        expr_type = type(expr).__name__

        if expr_type == "Var":
            if expr.name in self.env:
                return self.env[expr.name]
            raise ValueError(f"Unbound variable {expr.name}")

        elif expr_type in ("Literal", "LiteralWrapper"):
            val = expr.value
            if isinstance(val, float):
                f = self.next_f32()
                self.emit(f"    mov.f32 {f}, {float_to_ptx_hex(val)};")
                return f
            else:
                r = self.next_b32()
                self.emit(f"    mov.u32 {r}, {int(val)};")
                return r

        elif expr_type in ("BinaryExpr", "BinaryExprWrapper"):
            op = expr.op
            out_reg = self.next_b32()

            # Check if right operand is an integer literal for immediate arithmetic
            if is_lit(expr.right) and not isinstance(get_lit(expr.right), float):
                left_reg = self.compile_expr(expr.left)
                imm = int(get_lit(expr.right))
                if op == "+":
                    self.emit(f"    add.s32 {out_reg}, {left_reg}, {imm};")
                elif op == "*":
                    self.emit(f"    mul.lo.s32 {out_reg}, {left_reg}, {imm};")
                elif op == "-":
                    self.emit(f"    sub.s32 {out_reg}, {left_reg}, {imm};")
                elif op == "/":
                    self.emit(f"    div.s32 {out_reg}, {left_reg}, {imm};")
                elif op == "%":
                    self.emit(f"    rem.s32 {out_reg}, {left_reg}, {imm};")
                elif op in ("==", "!=", "<", "<=", ">", ">="):
                    p_reg = self.compile_cond(expr)
                    r_one = self.next_b32()
                    r_zero = self.next_b32()
                    self.emit(f"    mov.u32 {r_one}, 1;")
                    self.emit(f"    mov.u32 {r_zero}, 0;")
                    self.emit(f"    selp.b32 {out_reg}, {r_one}, {r_zero}, {p_reg};")
                else:
                    raise NotImplementedError(f"Unsupported binary expr op: {op}")
                return out_reg

            left_reg = self.compile_expr(expr.left)
            right_reg = self.compile_expr(expr.right)
            if op == "+":
                self.emit(f"    add.s32 {out_reg}, {left_reg}, {right_reg};")
            elif op == "*":
                self.emit(f"    mul.lo.s32 {out_reg}, {left_reg}, {right_reg};")
            elif op == "-":
                self.emit(f"    sub.s32 {out_reg}, {left_reg}, {right_reg};")
            elif op == "/":
                self.emit(f"    div.s32 {out_reg}, {left_reg}, {right_reg};")
            elif op == "%":
                self.emit(f"    rem.s32 {out_reg}, {left_reg}, {right_reg};")
            elif op in ("==", "!=", "<", "<=", ">", ">="):
                p_reg = self.compile_cond(expr)
                r_one = self.next_b32()
                r_zero = self.next_b32()
                self.emit(f"    mov.u32 {r_one}, 1;")
                self.emit(f"    mov.u32 {r_zero}, 0;")
                self.emit(f"    selp.b32 {out_reg}, {r_one}, {r_zero}, {p_reg};")
            else:
                raise NotImplementedError(f"Unsupported binary expr op: {op}")
            return out_reg

        elif expr_type == "TernaryExpr":
            p_cond = self.compile_cond(expr.cond)
            r_true = self.compile_expr(expr.if_true)
            r_false = self.compile_expr(expr.if_false)
            out_reg = self.next_b32()
            self.emit(f"    selp.b32 {out_reg}, {r_true}, {r_false}, {p_cond};")
            return out_reg

        else:
            raise NotImplementedError(f"Unsupported expr type: {expr_type}")

    def compile_cond(self, cond) -> str:
        cond = simplify_expr(cond)
        cond_type = type(cond).__name__
        if cond_type in ("BinaryExpr", "BinaryExprWrapper"):
            left_reg = self.compile_expr(cond.left)
            op = cond.op
            p_reg = self.next_pred()

            if is_lit(cond.right) and not isinstance(get_lit(cond.right), float):
                imm = int(get_lit(cond.right))
                if op == "==":
                    self.emit(f"    setp.eq.s32 {p_reg}, {left_reg}, {imm};")
                elif op == "!=":
                    self.emit(f"    setp.ne.s32 {p_reg}, {left_reg}, {imm};")
                elif op == "<":
                    self.emit(f"    setp.lt.s32 {p_reg}, {left_reg}, {imm};")
                elif op == "<=":
                    self.emit(f"    setp.le.s32 {p_reg}, {left_reg}, {imm};")
                elif op == ">":
                    self.emit(f"    setp.gt.s32 {p_reg}, {left_reg}, {imm};")
                elif op == ">=":
                    self.emit(f"    setp.ge.s32 {p_reg}, {left_reg}, {imm};")
                else:
                    raise NotImplementedError(f"Unsupported condition op: {op}")
                return p_reg

            right_reg = self.compile_expr(cond.right)
            if op == "==":
                self.emit(f"    setp.eq.s32 {p_reg}, {left_reg}, {right_reg};")
            elif op == "!=":
                self.emit(f"    setp.ne.s32 {p_reg}, {left_reg}, {right_reg};")
            elif op == "<":
                self.emit(f"    setp.lt.s32 {p_reg}, {left_reg}, {right_reg};")
            elif op == "<=":
                self.emit(f"    setp.le.s32 {p_reg}, {left_reg}, {right_reg};")
            elif op == ">":
                self.emit(f"    setp.gt.s32 {p_reg}, {left_reg}, {right_reg};")
            elif op == ">=":
                self.emit(f"    setp.ge.s32 {p_reg}, {left_reg}, {right_reg};")
            else:
                raise NotImplementedError(f"Unsupported condition op: {op}")
            return p_reg
        else:
            raise NotImplementedError(f"Unsupported cond type: {cond_type}")

    # ==========================================================================
    # PASS 2: Address Strength Reduction (mad.wide.s32 & mad.lo.s32)
    # ==========================================================================
    def compile_global_address(self, buf_name: str, index_exprs: tuple | list) -> str:
        """Compute 64-bit global memory address directly using mad.wide.s32."""
        base_ptr = self.buf_ptrs[buf_name]
        if len(index_exprs) == 0:
            return base_ptr
        elif len(index_exprs) == 1:
            idx_reg = self.compile_expr(index_exprs[0])
            addr_reg = self.next_b64()
            self.emit(f"    mad.wide.s32 {addr_reg}, {idx_reg}, 4, {base_ptr};")
            return addr_reg
        else:
            shape = self.buf_shapes.get(buf_name, [])
            strides = [1] * len(index_exprs)
            for i in range(len(index_exprs) - 2, -1, -1):
                extent = shape[i + 1] if i + 1 < len(shape) else 1
                strides[i] = strides[i + 1] * int(extent)

            total_idx = None
            for expr, stride in zip(index_exprs, strides):
                simplified = simplify_expr(expr)
                if is_lit(simplified, 0):
                    continue
                idx_part = self.compile_expr(simplified)
                if stride != 1:
                    scaled = self.next_b32()
                    self.emit(f"    mul.lo.s32 {scaled}, {idx_part}, {stride};")
                    idx_part = scaled

                if total_idx is None:
                    total_idx = idx_part
                else:
                    next_total = self.next_b32()
                    self.emit(f"    add.s32 {next_total}, {total_idx}, {idx_part};")
                    total_idx = next_total

            if total_idx is None:
                return base_ptr

            addr_reg = self.next_b64()
            self.emit(f"    mad.wide.s32 {addr_reg}, {total_idx}, 4, {base_ptr};")
            return addr_reg

    def compile_shared_address(self, buf_name: str, index_exprs: tuple | list) -> str:
        """Compute 32-bit shared memory address directly using mad.lo.s32."""
        base_addr = self.smem_bases[buf_name]
        if len(index_exprs) == 0:
            return base_addr
        elif len(index_exprs) == 1:
            idx_reg = self.compile_expr(index_exprs[0])
            addr_reg = self.next_b32()
            self.emit(f"    mad.lo.s32 {addr_reg}, {idx_reg}, 4, {base_addr};")
            return addr_reg
        else:
            shape = self.buf_shapes.get(buf_name, [])
            strides = [1] * len(index_exprs)
            for i in range(len(index_exprs) - 2, -1, -1):
                extent = shape[i + 1] if i + 1 < len(shape) else 1
                strides[i] = strides[i + 1] * int(extent)

            total_idx = None
            for expr, stride in zip(index_exprs, strides):
                simplified = simplify_expr(expr)
                if is_lit(simplified, 0):
                    continue
                idx_part = self.compile_expr(simplified)
                if stride != 1:
                    scaled = self.next_b32()
                    self.emit(f"    mul.lo.s32 {scaled}, {idx_part}, {stride};")
                    idx_part = scaled

                if total_idx is None:
                    total_idx = idx_part
                else:
                    next_total = self.next_b32()
                    self.emit(f"    add.s32 {next_total}, {total_idx}, {idx_part};")
                    total_idx = next_total

            if total_idx is None:
                return base_addr

            addr_reg = self.next_b32()
            self.emit(f"    mad.lo.s32 {addr_reg}, {total_idx}, 4, {base_addr};")
            return addr_reg


def find_smem_decls(stmts) -> list[tuple[str, int, int, list[int]]]:
    """Recursively find Smem statements: returns list of (name, bytes, align, shape)."""
    res = []
    for s in stmts:
        if type(s).__name__ == "Smem":
            extents = [to_int(e) for e in s.extents] if s.extents else [1]
            n_elems = 1
            for e in extents:
                n_elems *= e
            n_bytes = n_elems * 4
            align = s.align if s.align else 16
            res.append((s.name, n_bytes, align, extents))
        if hasattr(s, "body"):
            res.extend(find_smem_decls(s.body))
        if hasattr(s, "else_body"):
            res.extend(find_smem_decls(s.else_body))
    return res


def compile_kernel(ir_path: Path) -> str:
    with open(ir_path) as f:
        data = json.load(f)

    g = Graph.from_dict(data)

    meta_path = ir_path.parent / "meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    kop_node = None
    for name, node in g.nodes.items():
        if hasattr(node.op, "name") and hasattr(node.op, "body"):
            kop_node = node
            break

    if kop_node is None:
        raise ValueError(f"No KernelOp found in {ir_path}")

    kop = kop_node.op
    entry_name = kop.name

    arg_order: list[str] = []
    if meta:
        for k in meta.get("kernels", []):
            if k.get("entry") == entry_name:
                arg_order = k.get("arg_order", [])
                break

    if not arg_order:
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        arg_order = list(inputs) + [o for o in outputs if o not in inputs]

    emitter = PTXEmitter()

    for bname, bnode in g.nodes.items():
        if hasattr(bnode, "outputs") and bnode.outputs:
            out_obj = bnode.outputs[0]
            if hasattr(out_obj, "shape"):
                emitter.buf_shapes[bname] = [to_int(d) for d in out_obj.shape]

    smem_decls = find_smem_decls(kop.body)
    total_smem_bytes = sum(b for _, b, _, _ in smem_decls)
    smem_bytes = 0
    if meta:
        for k in meta.get("kernels", []):
            if k.get("entry") == entry_name:
                smem_bytes = k.get("smem_bytes", 0)
                break
    use_dynamic_smem = (smem_bytes > 0) or (total_smem_bytes > 48 * 1024)

    cur_offset = 0
    smem_offsets: dict[str, int] = {}
    for s_name, s_bytes, s_align, s_shape in smem_decls:
        emitter.smem_names.add(s_name)
        emitter.buf_shapes[s_name] = s_shape
        if cur_offset % s_align != 0:
            cur_offset += s_align - (cur_offset % s_align)
        smem_offsets[s_name] = cur_offset
        cur_offset += s_bytes

    # --- Prologue ---
    emitter.emit(".version 8.0")
    emitter.emit(".target sm_89")
    emitter.emit(".address_size 64")
    emitter.emit("")

    if use_dynamic_smem:
        emitter.emit(".extern .shared .align 16 .b8 _smem_pool[];")
        emitter.emit("")
    else:
        for s_name, s_bytes, s_align, _ in smem_decls:
            emitter.emit(f".shared .align {s_align} .b8 {s_name}[{s_bytes}];")
        if smem_decls:
            emitter.emit("")

    emitter.emit(f".visible .entry {entry_name}(")
    for i, arg in enumerate(arg_order):
        comma = "," if i < len(arg_order) - 1 else ""
        emitter.emit(f"    .param .u64 param_{arg}{comma}")
    emitter.emit(")")
    emitter.emit("{")

    body_emitter = PTXEmitter()
    body_emitter.buf_shapes = emitter.buf_shapes
    body_emitter.smem_names = emitter.smem_names
    body_emitter.smem_bases = {}

    if use_dynamic_smem:
        r_pool = body_emitter.next_b32()
        body_emitter.emit(f"    mov.u32 {r_pool}, _smem_pool;")
        for s_name, s_off in smem_offsets.items():
            if s_off == 0:
                body_emitter.smem_bases[s_name] = r_pool
            else:
                r_base = body_emitter.next_b32()
                body_emitter.emit(f"    add.s32 {r_base}, {r_pool}, {s_off};")
                body_emitter.smem_bases[s_name] = r_base
    else:
        for s_name in smem_offsets:
            r_base = body_emitter.next_b32()
            body_emitter.emit(f"    mov.u32 {r_base}, {s_name};")
            body_emitter.smem_bases[s_name] = r_base

    for arg in arg_order:
        p_reg = body_emitter.next_b64()
        g_reg = body_emitter.next_b64()
        body_emitter.emit(f"    ld.param.u64 {p_reg}, [param_{arg}];")
        body_emitter.emit(f"    cvta.to.global.u64 {g_reg}, {p_reg};")
        body_emitter.buf_ptrs[arg] = g_reg

    r_tid = body_emitter.next_b32()
    r_ctaid = body_emitter.next_b32()
    r_ntid = body_emitter.next_b32()
    r_gid = body_emitter.next_b32()
    body_emitter.emit(f"    mov.u32 {r_tid}, %tid.x;")
    body_emitter.emit(f"    mov.u32 {r_ctaid}, %ctaid.x;")
    body_emitter.emit(f"    mov.u32 {r_ntid}, %ntid.x;")
    body_emitter.emit(f"    mad.lo.u32 {r_gid}, {r_ctaid}, {r_ntid}, {r_tid};")

    r_lane = body_emitter.next_b32()
    r_warp = body_emitter.next_b32()
    body_emitter.emit(f"    and.b32 {r_lane}, {r_tid}, 31;")
    body_emitter.emit(f"    shr.u32 {r_warp}, {r_tid}, 5;")
    body_emitter.env["lane"] = r_lane
    body_emitter.env["warp"] = r_warp

    exit_label = "EXIT"

    for stmt in kop.body:
        stype = type(stmt).__name__
        if stype == "Tile":
            extents = [to_int(a.extent) for a in stmt.axes]
            total_n = 1
            for e in extents:
                total_n *= e

            p_exit = body_emitter.next_pred()
            body_emitter.emit(f"    setp.ge.s32 {p_exit}, {r_gid}, {total_n};")
            body_emitter.emit(f"    @{p_exit} bra {exit_label};")

            strides = [1] * len(extents)
            for i in range(len(extents) - 2, -1, -1):
                strides[i] = strides[i + 1] * extents[i + 1]

            for i, a in enumerate(stmt.axes):
                r_div = body_emitter.next_b32()
                if strides[i] != 1:
                    body_emitter.emit(f"    div.s32 {r_div}, {r_gid}, {strides[i]};")
                else:
                    r_div = r_gid
                if i > 0:
                    r_rem = body_emitter.next_b32()
                    body_emitter.emit(f"    rem.s32 {r_rem}, {r_div}, {extents[i]};")
                    body_emitter.env[a.name] = r_rem
                else:
                    body_emitter.env[a.name] = r_div

            compile_stmts(stmt.body, body_emitter, exit_label)

    body_emitter.emit(f"{exit_label}:")
    body_emitter.emit("    ret;")
    body_emitter.emit("}")

    all_lines = emitter.lines[:]
    all_lines.append(f"    .reg .pred %p<{max(body_emitter.pred_count + 1, 4)}>;")
    all_lines.append(f"    .reg .f32  %f<{max(body_emitter.f32_count + 1, 16)}>;")
    all_lines.append(f"    .reg .b32  %r<{max(body_emitter.b32_count + 1, 16)}>;")
    all_lines.append(f"    .reg .b64  %rd<{max(body_emitter.b64_count + 1, 8)}>;")
    all_lines.append("")
    all_lines.extend(body_emitter.lines)

    return "\n".join(all_lines) + "\n"


def find_axis_accums(axis_name: str, stmts, emitter: PTXEmitter) -> list:
    res = []
    for s in stmts:
        if type(s).__name__ == "Accum":
            if (
                hasattr(s, "axes")
                and axis_name in s.axes
                and s.name not in emitter.init_names
            ):
                res.append(s)
        if hasattr(s, "body"):
            res.extend(find_axis_accums(axis_name, s.body, emitter))
    return res


def compile_stmts(stmts, emitter: PTXEmitter, exit_label: str):
    for stmt in stmts:
        stype = type(stmt).__name__

        if stype == "Smem":
            pass

        elif stype == "Sync":
            emitter.emit("    bar.sync 0;")

        elif stype == "Init":
            emitter.init_names.add(stmt.name)
            ident = stmt.identity.value if hasattr(stmt.identity, "value") else stmt.identity
            f_reg = emitter.next_f32()
            emitter.emit(f"    mov.f32 {f_reg}, {float_to_ptx_hex(ident)};")
            emitter.env[stmt.name] = f_reg

        elif stype == "Let":
            dtype_str = stmt.dtype.name if hasattr(stmt.dtype, "name") else str(stmt.dtype)
            val = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
            if dtype_str == "f32" or isinstance(val, float):
                f_reg = emitter.next_f32()
                emitter.emit(f"    mov.f32 {f_reg}, {float_to_ptx_hex(val)};")
                emitter.env[stmt.name] = f_reg
            else:
                r_reg = emitter.next_b32()
                emitter.emit(f"    mov.u32 {r_reg}, {int(val)};")
                emitter.env[stmt.name] = r_reg

        elif stype == "Load":
            buf_name = stmt.input
            if buf_name in emitter.smem_names:
                addr = emitter.compile_shared_address(buf_name, stmt.index)
                if stmt.width == 1:
                    f_reg = emitter.next_f32()
                    emitter.emit(f"    ld.shared.f32 {f_reg}, [{addr}];")
                    emitter.env[stmt.names[0]] = f_reg
                elif stmt.width == 2:
                    f_reg0 = emitter.next_f32()
                    f_reg1 = emitter.next_f32()
                    emitter.emit(f"    ld.shared.v2.f32 {{{f_reg0}, {f_reg1}}}, [{addr}];")
                    emitter.env[stmt.names[0]] = f_reg0
                    emitter.env[stmt.names[1]] = f_reg1
                else:
                    raise NotImplementedError(f"Shared vector load width {stmt.width}")
            else:
                addr = emitter.compile_global_address(buf_name, stmt.index)
                if stmt.width == 1:
                    f_reg = emitter.next_f32()
                    emitter.emit(f"    ld.global.f32 {f_reg}, [{addr}];")
                    emitter.env[stmt.names[0]] = f_reg
                elif stmt.width == 2:
                    f_reg0 = emitter.next_f32()
                    f_reg1 = emitter.next_f32()
                    emitter.emit(f"    ld.global.v2.f32 {{{f_reg0}, {f_reg1}}}, [{addr}];")
                    emitter.env[stmt.names[0]] = f_reg0
                    emitter.env[stmt.names[1]] = f_reg1
                elif stmt.width == 4:
                    f_regs = [emitter.next_f32() for _ in range(4)]
                    emitter.emit(f"    ld.global.v4.f32 {{{', '.join(f_regs)}}}, [{addr}];")
                    for name, r in zip(stmt.names, f_regs):
                        emitter.env[name] = r
                else:
                    raise NotImplementedError(f"Vector load width {stmt.width} not implemented yet")

        # ======================================================================
        # PASS 4: FMA (Fused Multiply-Add) Pattern Matching
        # ======================================================================
        elif stype == "Assign":
            op_name = stmt.op.name
            args = [emitter.env[a] for a in stmt.args]
            out_reg = emitter.next_f32()

            if op_name == "multiply":
                emitter.mul_defs[stmt.name] = (args[0], args[1])
                emitter.emit(f"    mul.f32 {out_reg}, {args[0]}, {args[1]};")
            elif op_name == "add":
                # FMA fusion: check if either operand is an in-flight multiply
                if stmt.args[0] in emitter.mul_defs:
                    ma, mb = emitter.mul_defs[stmt.args[0]]
                    emitter.emit(f"    fma.rn.f32 {out_reg}, {ma}, {mb}, {args[1]};")
                elif stmt.args[1] in emitter.mul_defs:
                    ma, mb = emitter.mul_defs[stmt.args[1]]
                    emitter.emit(f"    fma.rn.f32 {out_reg}, {ma}, {mb}, {args[0]};")
                else:
                    emitter.emit(f"    add.f32 {out_reg}, {args[0]}, {args[1]};")
            elif op_name == "divide":
                emitter.emit(f"    div.rn.f32 {out_reg}, {args[0]}, {args[1]};")
            elif op_name == "relu":
                emitter.emit(f"    max.f32 {out_reg}, {args[0]}, 0f00000000;")
            elif op_name == "maximum":
                emitter.emit(f"    max.f32 {out_reg}, {args[0]}, {args[1]};")
            elif op_name in ("sub", "subtract"):
                emitter.emit(f"    sub.f32 {out_reg}, {args[0]}, {args[1]};")
            elif op_name == "reciprocal":
                emitter.emit(f"    rcp.approx.f32 {out_reg}, {args[0]};")
            elif op_name == "sqrt":
                emitter.emit(f"    sqrt.approx.f32 {out_reg}, {args[0]};")
            elif op_name == "exp":
                f_scaled = emitter.next_f32()
                emitter.emit(f"    mul.f32 {f_scaled}, {args[0]}, 0f3fb8aa3b;")
                emitter.emit(f"    ex2.approx.f32 {out_reg}, {f_scaled};")
            elif op_name == "copy":
                emitter.emit(f"    mov.f32 {out_reg}, {args[0]};")
            else:
                raise NotImplementedError(f"Unsupported assign op: {op_name}")
            emitter.env[stmt.name] = out_reg

        elif stype == "Accum":
            val_reg = emitter.env[stmt.value]
            acc_name = stmt.name
            if acc_name not in emitter.env:
                acc_reg = emitter.next_f32()
                ident = stmt.op.identity if hasattr(stmt.op, "identity") else 0.0
                emitter.emit(f"    mov.f32 {acc_reg}, {float_to_ptx_hex(ident)};")
                emitter.env[acc_name] = acc_reg
            else:
                acc_reg = emitter.env[acc_name]

            lhs = emitter.env[stmt.base] if stmt.base is not None else acc_reg
            rhs = val_reg

            op_name = stmt.op.name
            if op_name == "add":
                # FMA Fusion into accumulator!
                if stmt.value in emitter.mul_defs:
                    ma, mb = emitter.mul_defs[stmt.value]
                    emitter.emit(f"    fma.rn.f32 {acc_reg}, {ma}, {mb}, {lhs};")
                else:
                    emitter.emit(f"    add.f32 {acc_reg}, {lhs}, {rhs};")
            elif op_name == "maximum":
                emitter.emit(f"    max.f32 {acc_reg}, {lhs}, {rhs};")
            else:
                raise NotImplementedError(f"Unsupported accum op: {op_name}")

        elif stype == "StridedLoop":
            for acc in find_axis_accums(stmt.axis.name, stmt.body, emitter):
                acc_reg = emitter.next_f32()
                ident = acc.op.identity if hasattr(acc.op, "identity") else 0.0
                emitter.emit(f"    mov.f32 {acc_reg}, {float_to_ptx_hex(ident)};")
                emitter.env[acc.name] = acc_reg

            start_reg = emitter.compile_expr(stmt.start)
            r_loop = emitter.next_b32()
            emitter.emit(f"    mov.u32 {r_loop}, {start_reg};")
            emitter.env[stmt.axis.name] = r_loop

            loop_start = emitter.next_label("LOOP_START")
            loop_end = emitter.next_label("LOOP_END")
            p_loop = emitter.next_pred()
            extent = to_int(stmt.axis.extent)
            step_val = to_int(stmt.step.value)

            emitter.emit(f"{loop_start}:")
            emitter.emit(f"    setp.ge.s32 {p_loop}, {r_loop}, {extent};")
            emitter.emit(f"    @{p_loop} bra {loop_end};")

            compile_stmts(stmt.body, emitter, exit_label)

            emitter.emit(f"    add.s32 {r_loop}, {r_loop}, {step_val};")
            emitter.emit(f"    bra {loop_start};")
            emitter.emit(f"{loop_end}:")

        elif stype == "Loop":
            for acc in find_axis_accums(stmt.axis.name, stmt.body, emitter):
                acc_reg = emitter.next_f32()
                ident = acc.op.identity if hasattr(acc.op, "identity") else 0.0
                emitter.emit(f"    mov.f32 {acc_reg}, {float_to_ptx_hex(ident)};")
                emitter.env[acc.name] = acc_reg

            r_loop = emitter.next_b32()
            emitter.emit(f"    mov.u32 {r_loop}, 0;")
            emitter.env[stmt.axis.name] = r_loop

            loop_start = emitter.next_label("LOOP_START")
            loop_end = emitter.next_label("LOOP_END")
            p_loop = emitter.next_pred()
            extent = to_int(stmt.axis.extent)

            emitter.emit(f"{loop_start}:")
            emitter.emit(f"    setp.ge.s32 {p_loop}, {r_loop}, {extent};")
            emitter.emit(f"    @{p_loop} bra {loop_end};")

            compile_stmts(stmt.body, emitter, exit_label)

            emitter.emit(f"    add.s32 {r_loop}, {r_loop}, 1;")
            emitter.emit(f"    bra {loop_start};")
            emitter.emit(f"{loop_end}:")

        elif stype == "CpAsyncCopy":
            total_smem_index = list(stmt.smem_index)
            if hasattr(stmt, "lane_index") and stmt.lane_index is not None:
                total_smem_index = [
                    BinaryExprWrapper("+", s, l) for s, l in zip(stmt.smem_index, stmt.lane_index)
                ]

            r_smem_addr = emitter.compile_shared_address(stmt.smem, total_smem_index)
            rd_gmem_addr = emitter.compile_global_address(stmt.src, stmt.src_index)
            emitter.emit(f"    cp.async.ca.shared.global [{r_smem_addr}], [{rd_gmem_addr}], {stmt.nbytes};")

        elif stype == "CpAsyncCommit":
            emitter.emit("    cp.async.commit_group;")

        elif stype == "CpAsyncWait":
            group = getattr(stmt, "group", 0)
            emitter.emit(f"    cp.async.wait_group {group};")

        elif stype == "WarpShuffle":
            state = stmt.state
            state_b = getattr(stmt, "state_b", tuple(f"{s}__o" for s in state))
            combine_states = getattr(stmt, "combine_states", None)

            for delta in [16, 8, 4, 2, 1]:
                for s_name, sb_name in zip(state, state_b):
                    curr_reg = emitter.env[s_name]
                    r_tmp = emitter.next_b32()
                    r_shfl = emitter.next_b32()
                    f_shfl = emitter.next_f32()
                    emitter.emit(f"    mov.b32 {r_tmp}, {curr_reg};")
                    emitter.emit(f"    shfl.sync.bfly.b32 {r_shfl}, {r_tmp}, {delta}, 0x1f, 0xffffffff;")
                    emitter.emit(f"    mov.b32 {f_shfl}, {r_shfl};")
                    emitter.env[sb_name] = f_shfl

                if combine_states:
                    compile_stmts(combine_states, emitter, exit_label)
                else:
                    for s_name, sb_name in zip(state, state_b):
                        next_reg = emitter.next_f32()
                        emitter.emit(f"    add.f32 {next_reg}, {emitter.env[s_name]}, {emitter.env[sb_name]};")
                        emitter.env[s_name] = next_reg

        elif stype == "TreeHalve":
            state = stmt.state
            state_b = getattr(stmt, "state_b", tuple(f"{s}__o" for s in state))
            bufs = stmt.bufs
            combine_states = getattr(stmt, "combine_states", None)
            r_warp = emitter.env["warp"]
            length = stmt.length

            s = length // 2
            while s > 0:
                p_s = emitter.next_pred()
                l_after = emitter.next_label(f"AFTER_TH_{s}")
                emitter.emit(f"    setp.lt.s32 {p_s}, {r_warp}, {s};")
                emitter.emit(f"    @!{p_s} bra {l_after};")

                for buf_name, s_name, sb_name in zip(bufs, state, state_b):
                    r_base = emitter.smem_bases[buf_name]
                    r_off1 = emitter.next_b32()
                    r_addr1 = emitter.next_b32()
                    f_v1 = emitter.next_f32()
                    emitter.emit(f"    shl.b32 {r_off1}, {r_warp}, 2;")
                    emitter.emit(f"    add.s32 {r_addr1}, {r_base}, {r_off1};")
                    emitter.emit(f"    ld.shared.f32 {f_v1}, [{r_addr1}];")
                    emitter.env[s_name] = f_v1

                    r_warp_s = emitter.next_b32()
                    r_off2 = emitter.next_b32()
                    r_addr2 = emitter.next_b32()
                    f_v2 = emitter.next_f32()
                    emitter.emit(f"    add.s32 {r_warp_s}, {r_warp}, {s};")
                    emitter.emit(f"    shl.b32 {r_off2}, {r_warp_s}, 2;")
                    emitter.emit(f"    add.s32 {r_addr2}, {r_base}, {r_off2};")
                    emitter.emit(f"    ld.shared.f32 {f_v2}, [{r_addr2}];")
                    emitter.env[sb_name] = f_v2

                if combine_states:
                    compile_stmts(combine_states, emitter, exit_label)
                else:
                    for s_name, sb_name in zip(state, state_b):
                        next_reg = emitter.next_f32()
                        emitter.emit(f"    add.f32 {next_reg}, {emitter.env[s_name]}, {emitter.env[sb_name]};")
                        emitter.env[s_name] = next_reg

                for buf_name, s_name in zip(bufs, state):
                    r_base = emitter.smem_bases[buf_name]
                    r_off1 = emitter.next_b32()
                    r_addr1 = emitter.next_b32()
                    emitter.emit(f"    shl.b32 {r_off1}, {r_warp}, 2;")
                    emitter.emit(f"    add.s32 {r_addr1}, {r_base}, {r_off1};")
                    emitter.emit(f"    st.shared.f32 [{r_addr1}], {emitter.env[s_name]};")

                emitter.emit(f"{l_after}:")
                emitter.emit("    bar.sync 0;")
                s //= 2

            for buf_name, s_name in zip(bufs, state):
                r_base0 = emitter.smem_bases[buf_name]
                f_final = emitter.next_f32()
                emitter.emit(f"    ld.shared.f32 {f_final}, [{r_base0}];")
                emitter.env[s_name] = f_final

        elif stype == "Cond":
            p_cond = emitter.compile_cond(stmt.cond)
            l_else = emitter.next_label("L_ELSE")
            l_after = emitter.next_label("L_AFTER_COND")

            if stmt.else_body:
                emitter.emit(f"    @!{p_cond} bra {l_else};")
                compile_stmts(stmt.body, emitter, exit_label)
                emitter.emit(f"    bra {l_after};")
                emitter.emit(f"{l_else}:")
                compile_stmts(stmt.else_body, emitter, exit_label)
                emitter.emit(f"{l_after}:")
            else:
                emitter.emit(f"    @!{p_cond} bra {l_after};")
                compile_stmts(stmt.body, emitter, exit_label)
                emitter.emit(f"{l_after}:")

        elif stype == "Write":
            buf_name = stmt.output
            val_regs = [emitter.env[v] for v in stmt.values]

            if buf_name in emitter.smem_names:
                addr = emitter.compile_shared_address(buf_name, stmt.index)
                emitter.emit(f"    st.shared.f32 [{addr}], {val_regs[0]};")
            else:
                addr = emitter.compile_global_address(buf_name, stmt.index)
                is_atomic = getattr(stmt, "atomic", False)
                if is_atomic:
                    for val_reg in val_regs:
                        emitter.emit(f"    red.global.add.f32 [{addr}], {val_reg};")
                else:
                    if len(val_regs) == 1:
                        emitter.emit(f"    st.global.f32 [{addr}], {val_regs[0]};")
                    elif len(val_regs) == 2:
                        emitter.emit(f"    st.global.v2.f32 [{addr}], {{{val_regs[0]}, {val_regs[1]}}};")
                    elif len(val_regs) == 4:
                        emitter.emit(
                            f"    st.global.v4.f32 [{addr}], {{{val_regs[0]}, {val_regs[1]}, {val_regs[2]}, {val_regs[3]}}};"
                        )
                    else:
                        raise NotImplementedError(f"Unsupported write width: {len(val_regs)}")

        else:
            raise NotImplementedError(f"Unsupported stmt type: {stype}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compile.py path/to/<kernel>.ir.json", file=sys.stderr)
        sys.exit(1)

    ir_path = Path(sys.argv[1])
    ptx = compile_kernel(ir_path)
    sys.stdout.write(ptx)


if __name__ == "__main__":
    main()
