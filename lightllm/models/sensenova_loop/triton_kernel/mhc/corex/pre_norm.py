# SPDX-License-Identifier: Apache-2.0

"""CoreX mHC pre-norm.

投影 kernel 与 MetaX 相同：M 是编译期常量，tl.range 扫 K。
MR-V100 上 2 stream（M=8）且 tokens <= 16 时 split-K 不慢于 fp32 GEMM，更长用 PyTorch。
4 stream 的 M=24 不是 2 的幂，这台 Triton 的 tl.zeros((M,)) 编不过。
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from lightllm.models.sensenova_loop.triton_kernel.mhc.pre_norm import (
    _hc_pre_combine_norm_kernel,
    _hc_prepare_prenorm_kernel,
)


def is_corex_mhc() -> bool:
    try:
        from lightllm.utils.envs_utils import get_env_start_args

        return getattr(get_env_start_args(), "hardware_platform", "cuda") == "corex"
    except Exception:
        return False


@triton.jit
def _hc_proj_corex_kernel(
    x,
    fn,
    gemm_partial,
    sqrsum_partial,
    stride_x,
    stride_fn,
    gemm_stride_s,
    gemm_stride_m,
    sqrsum_stride_s,
    K,
    M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    split = tl.program_id(1)
    k_per = tl.cdiv(K, tl.num_programs(1))
    start = split * k_per
    acc = tl.zeros((M,), dtype=tl.float32)
    sq = 0.0
    mix_offsets = tl.arange(0, M)
    for k0 in tl.range(0, k_per, BLOCK_K):
        k_offsets = start + k0 + tl.arange(0, BLOCK_K)
        mask = (k_offsets < start + k_per) & (k_offsets < K)
        values = tl.load(x + token * stride_x + k_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(fn + mix_offsets[:, None] * stride_fn + k_offsets[None, :], mask=mask[None, :], other=0.0)
        acc += tl.sum(weight * values[None, :], axis=1)
        sq += tl.sum(values * values, axis=0)
    tl.store(gemm_partial + split * gemm_stride_s + token * gemm_stride_m + mix_offsets, acc)
    tl.store(sqrsum_partial + split * sqrsum_stride_s + token, sq)


def hc_pre_norm_corex(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    streams: int,
    rms_eps: float,
    norm_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[-1] % streams == 0
    assert streams in (2, 4)
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32) and fn.dtype == torch.float32
    assert x.is_contiguous() and fn.is_contiguous() and norm_weight.is_contiguous()
    assert fn.shape == (2 * streams + streams * streams, x.shape[1])
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    mix_size = fn.shape[0]
    if streams == 2 and tokens <= 16:
        n_splits = 16
        gemm_partial = torch.empty((n_splits, tokens, mix_size), dtype=torch.float32, device=x.device)
        sqrsum_partial = torch.empty((n_splits, tokens), dtype=torch.float32, device=x.device)
        _hc_proj_corex_kernel[(tokens, n_splits)](
            x,
            fn,
            gemm_partial,
            sqrsum_partial,
            x.stride(0),
            fn.stride(0),
            gemm_partial.stride(0),
            gemm_partial.stride(1),
            sqrsum_partial.stride(0),
            flattened_hidden,
            M=mix_size,
            BLOCK_K=128,
            num_warps=4,
        )
    else:
        x_fp32 = x.float()
        gemm_partial = (x_fp32 @ fn.T).unsqueeze(0)
        sqrsum_partial = x_fp32.square().sum(dim=-1).unsqueeze(0)
        n_splits = 1

    pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    residual_mix = torch.empty((tokens, streams, streams), dtype=torch.float32, device=x.device)
    _hc_prepare_prenorm_kernel[(tokens,)](
        gemm_partial,
        sqrsum_partial,
        base,
        pre,
        post,
        residual_mix,
        gemm_partial.stride(0),
        gemm_partial.stride(1),
        sqrsum_partial.stride(0),
        sqrsum_partial.stride(1),
        pre.stride(0),
        residual_mix.stride(0),
        FLATTENED_HIDDEN=flattened_hidden,
        RMS_EPS=rms_eps,
        STREAMS=streams,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        num_warps=1,
    )

    layer_input = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _hc_pre_combine_norm_kernel[(tokens,)](
        x,
        pre,
        norm_weight,
        layer_input,
        hidden=hidden,
        x_stride_m=x.stride(0),
        pre_stride_m=pre.stride(0),
        out_stride_m=layer_input.stride(0),
        STREAMS=streams,
        NORM_EPS=norm_eps,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return layer_input, residual_mix, post
