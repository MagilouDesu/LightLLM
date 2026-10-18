import torch
from typing import Any, Callable, Optional, Tuple

import torch_mlu_ops as tmo

from lightllm.common.basemodel.triton_kernel.multimodal_emb import torch_multimodal_emb
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm_mlu import mlu_gated_rmsnorm_forward
from lightllm.platform.base.ops import register_op, out_like
from lightllm.server.embed_cache.copy_to_cache import torch_offload_embed_tensor_to_cache


@register_op("mlu")
def multimodal_emb(
    *,
    out: torch.Tensor,
    prompt_ids: torch.Tensor,
    text_weight_embs: torch.Tensor,
    embed_cache: torch.Tensor,
    img_token_lens: torch.Tensor,
    img_start_token_ids: torch.Tensor,
    img_start_locs_in_cache: torch.Tensor,
    tp_text_start_token_id: int,
    tp_text_end_token_id: int,
    tp_world_size: int,
) -> None:
    torch_multimodal_emb(
        out=out,
        prompt_ids=prompt_ids,
        text_weight_embs=text_weight_embs,
        embed_cache=embed_cache,
        img_token_lens=img_token_lens,
        img_start_token_ids=img_start_token_ids,
        img_start_locs_in_cache=img_start_locs_in_cache,
        tp_text_start_token_id=tp_text_start_token_id,
        tp_text_end_token_id=tp_text_end_token_id,
        tp_world_size=tp_world_size,
    )


@register_op("mlu")
def offload_embed_tensor_to_cache(
    *,
    embed_tensor: torch.Tensor,
    cache_tensor: torch.Tensor,
    start_index_in_cache: int,
) -> None:
    torch_offload_embed_tensor_to_cache(
        embed_tensor=embed_tensor,
        cache_tensor=cache_tensor,
        start_index_in_cache=start_index_in_cache,
    )


@register_op("mlu", out=out_like("input"))
def rms_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if gate_value is None:
        # vLLM RMSNorm.forward_oot. Gated norm has no vendor kernel.
        tmo.fused_rms_norm(input, None, weight, None, None, eps, False, None, out)
    else:
        mlu_gated_rmsnorm_forward(x=input, weight=weight, bias=None, eps=eps, z=gate_value, out=out)
    return out


@register_op("mlu")
def qk_rms_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    eps: float,
    fp32_multiply: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del fp32_multiply
    head_dim = w_q.shape[0]
    q.copy_(tmo.fused_rms_norm(q.reshape(-1, head_dim), None, w_q, None, None, eps, False).view_as(q))
    k.copy_(tmo.fused_rms_norm(k.reshape(-1, head_dim), None, w_k, None, None, eps, False).view_as(k))
    return q, k


@register_op("mlu")
def ffn(
    *,
    input: torch.Tensor,
    layer_weight: Any,
    alloc_func: Callable,
    embed_dim: int,
) -> torch.Tensor:
    input = input.view(-1, embed_dim)
    up_gate_out = layer_weight.gate_up_proj.mm(input)
    # tmo.active(..., is_gated=True) is silu(first half) * second half, matching silu_and_mul.
    ffn1_out = alloc_func(
        (input.size(0), up_gate_out.size(1) // 2),
        dtype=input.dtype,
        device=input.device,
    )
    torch.ops.torch_mlu_ops.active(up_gate_out, ffn1_out, None, None, "silu", True, 0, 0, 1.0, False, "none")
    return layer_weight.down_proj.mm(ffn1_out)
