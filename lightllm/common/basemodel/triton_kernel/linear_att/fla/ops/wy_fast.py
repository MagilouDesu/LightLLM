# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501

import torch

import triton
import triton.language as tl

from .index import prepare_chunk_indices
from lightllm.common.triton_utils.autotuner import autotune


def _wy_configs():
    return [{"num_warps": w, "num_stages": s} for w in (1, 2, 4, 8) for s in (1, 2, 3)]


def _wy_static(k, v, beta, g_cumsum, A, cu_seqlens):
    return {
        "H": int(v.shape[-2]),
        "K": int(k.shape[-1]),
        "V": int(v.shape[-1]),
        "BT": int(A.shape[-1]),
        "VARLEN": cu_seqlens is not None,
    }


def _wy_run(k, v, beta, g_cumsum, A, cu_seqlens):
    return int(k.shape[1])


@triton.jit
def _recompute_u_tile(
    v, u, b_A, b_beta, bos, i_h, i_t, i_v, T, H, V, BT: tl.constexpr, BV: tl.constexpr
):
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_u = tl.make_block_ptr(
        u + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
    b_u = tl.dot(b_A, b_vb, allow_tf32=False)
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _recompute_w_tile(
    k, w, b_A, b_beta, b_g, bos, i_h, i_t, i_k, T, H, Hg, K, BT: tl.constexpr, BK: tl.constexpr
):
    p_k = tl.make_block_ptr(
        k + (bos * Hg + i_h // (H // Hg)) * K,
        (T, K),
        (Hg * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_w = tl.make_block_ptr(
        w + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
    b_w = tl.dot(b_A, b_kb)
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    if NV <= 4:
        for i_v in tl.static_range(NV):
            _recompute_u_tile(v, u, b_A, b_beta, bos, i_h, i_t, i_v, T, H, V, BT, BV)
    else:
        for i_v in range(NV):
            _recompute_u_tile(v, u, b_A, b_beta, bos, i_h, i_t, i_v, T, H, V, BT, BV)
    if NK <= 4:
        for i_k in tl.static_range(NK):
            _recompute_w_tile(k, w, b_A, b_beta, b_g, bos, i_h, i_t, i_k, T, H, Hg, K, BT, BK)
    else:
        for i_k in range(NK):
            _recompute_w_tile(k, w, b_A, b_beta, b_g, bos, i_h, i_t, i_k, T, H, Hg, K, BT, BK)


@autotune(
    kernel_name="recompute_w_u_fwd:v1",
    configs_gen_func=_wy_configs,
    static_key_func=_wy_static,
    run_key_func=_wy_run,
)
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    run_config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    NV = triton.cdiv(V, BV)
    NK = triton.cdiv(K, BK)
    # Triton default when omitted is 4 warps / 3 stages. Keep that as the miss fallback.
    wy_cfg = {"num_warps": 4, "num_stages": 3} if run_config is None else run_config
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NV=NV,
        NK=NK,
        num_warps=int(wy_cfg["num_warps"]),
        num_stages=int(wy_cfg["num_stages"]),
    )
    return w, u
