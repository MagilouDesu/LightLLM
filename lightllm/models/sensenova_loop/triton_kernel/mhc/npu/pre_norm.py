from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from lightllm.models.sensenova_loop.triton_kernel.mhc.pre_norm import (
    LIGHTLLM_DISABLE_DEEPGEMM_MHC,
    _hc_pre_combine_norm_kernel,
    _hc_prepare_prenorm_kernel,
    hc_pre_norm,
)

_NPU_PROJ_HIDDEN = 5120
_NPU_PROJ_LAUNCH = {
    2: (128, 1024, 1, 2048),
    4: (32, 128, 4, 512),
}


def is_npu_mhc() -> bool:
    try:
        from lightllm.utils.envs_utils import get_env_start_args

        return getattr(get_env_start_args(), "hardware_platform", "cuda") == "ascend"
    except Exception:
        return hasattr(torch, "npu") and torch.npu.is_available()


@triton.jit
def _hc_proj_npu_kernel(
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
    """一次读完 bf16 残差，写出分段投影和平方和。

    grid 是 (tokens, splits)。每个 program 负责一个 token 的一段 K，沿 K 以 BLOCK_K 累加
    M 个 fp32 点积和平方和。sqrsum 的 token 维必须连续，存储下标直接用 token。
    """
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
        weight = tl.load(
            fn + mix_offsets[:, None] * stride_fn + k_offsets[None, :],
            mask=mask[None, :],
            other=0.0,
        )
        acc += tl.sum(weight * values[None, :], axis=1)
        sq += tl.sum(values * values, axis=0)
    tl.store(gemm_partial + split * gemm_stride_s + token * gemm_stride_m + mix_offsets, acc)
    tl.store(sqrsum_partial + split * sqrsum_stride_s + token, sq)


def _npu_proj_launch(streams: int, tokens: int, flattened_hidden: int):
    """返回 (splits, BLOCK_K)；这个形状不在更快的区间内时返回 None。"""
    if flattened_hidden != streams * _NPU_PROJ_HIDDEN:
        return None
    window = _NPU_PROJ_LAUNCH.get(streams)
    if window is None:
        return None
    low, high, splits, block_k = window
    if low <= tokens <= high:
        return splits, block_k
    return None


def _npu_proj_partials(x: torch.Tensor, fn: torch.Tensor, splits: int, block_k: int):
    tokens, flattened_hidden = x.shape
    mix_size = fn.shape[0]
    gemm_partial = torch.empty((splits, tokens, mix_size), dtype=torch.float32, device=x.device)
    sqrsum_partial = torch.empty((splits, tokens), dtype=torch.float32, device=x.device)
    _hc_proj_npu_kernel[(tokens, splits)](
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
        BLOCK_K=block_k,
    )
    return gemm_partial, sqrsum_partial


def hc_pre_norm_npu(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    streams: int,
    rms_eps: float,
    norm_eps: float,
    sinkhorn_iters: int,
    *,
    use_deepgemm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[-1] % streams == 0
    assert streams in (2, 4)
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32) and fn.dtype == torch.float32
    assert x.is_contiguous() and fn.is_contiguous() and norm_weight.is_contiguous()
    assert fn.shape == (2 * streams + streams * streams, x.shape[1])
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    use_pytorch_proj = (
        streams == 2 or not use_deepgemm or LIGHTLLM_DISABLE_DEEPGEMM_MHC or x.dtype != torch.bfloat16
    )
    launch = None
    if use_pytorch_proj and x.device.type == "npu" and x.dtype == torch.bfloat16:
        launch = _npu_proj_launch(streams, tokens, flattened_hidden)
    if launch is None:
        return hc_pre_norm(
            x,
            fn,
            base,
            norm_weight,
            streams,
            rms_eps,
            norm_eps,
            sinkhorn_iters,
            use_deepgemm=use_deepgemm,
        )

    n_splits, block_k = launch
    gemm_partial, sqrsum_partial = _npu_proj_partials(x, fn, n_splits, block_k)
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
