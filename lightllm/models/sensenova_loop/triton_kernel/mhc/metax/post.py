from __future__ import annotations

import torch
import triton

from lightllm.models.sensenova_loop.triton_kernel.mhc.post import _hc_post_kernel


def hc_post_metax(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    post_mix: torch.Tensor,
    streams: int,
) -> torch.Tensor:
    tokens, hidden = layer_output.shape
    assert streams in (2, 4)
    assert layer_output.is_contiguous() and residual.is_contiguous()
    output = torch.empty(
        (tokens, streams * hidden),
        dtype=layer_output.dtype,
        device=layer_output.device,
    )
    if streams == 2 and hidden == 4096:
        block_h, num_warps = 2048, 4
    else:
        block_h = min(triton.next_power_of_2(hidden), 1024)
        num_warps = 8
    _hc_post_kernel[(tokens, triton.cdiv(hidden, block_h))](
        layer_output,
        residual,
        residual_mix,
        post_mix,
        output,
        hidden=hidden,
        layer_stride_m=layer_output.stride(0),
        residual_stride_m=residual.stride(0),
        mix_stride_m=residual_mix.stride(0),
        mix_stride_i=residual_mix.stride(1),
        mix_stride_j=residual_mix.stride(2),
        post_stride_m=post_mix.stride(0),
        out_stride_m=output.stride(0),
        BLOCK_H=block_h,
        STREAMS=streams,
        num_warps=num_warps,
    )
    return output
