import torch
import torch.nn.functional as F


def mlu_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    from torch_mlu_ops import triton_ops

    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    if q_norm.shape[2] != v.shape[2]:
        repeat = v.shape[2] // q_norm.shape[2]
        q_norm = q_norm.repeat_interleave(repeat, dim=2).contiguous()
        k_norm = k_norm.repeat_interleave(repeat, dim=2).contiguous()
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(dtype=torch.int32)
    return triton_ops.chunk_gated_delta_rule(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
