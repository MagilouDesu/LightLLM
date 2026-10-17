import torch
from typing import Optional

from lightllm.common.basemodel.triton_kernel.multimodal_emb import torch_multimodal_emb
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm_mlu import mlu_gated_rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
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
        rmsnorm_forward(x=input, weight=weight, eps=eps, out=out)
    else:
        mlu_gated_rmsnorm_forward(x=input, weight=weight, bias=None, eps=eps, z=gate_value, out=out)
    return out
