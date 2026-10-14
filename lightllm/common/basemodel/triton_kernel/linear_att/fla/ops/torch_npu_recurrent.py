import torch
import torch.nn.functional as F

from .chunk_prefill_npu import _gather_state, _l2_qkv, _scatter_state, _seq_lengths


def torch_npu_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    num_accepted: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch_npu

    if q.shape[0] != 1:
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
    if write_indices.dim() == 1:
        write_indices = write_indices.view(-1, 1)
    write_indices = write_indices.contiguous()
    sequences, tokens = write_indices.shape
    if tokens > 8:
        raise ValueError(f"npu_recurrent_gated_delta_rule supports at most 8 tokens per sequence, got {tokens}")
    total = q.shape[1]
    key_dim = q.shape[-1]
    value_heads, value_dim = v.shape[2], v.shape[3]
    if cu_seqlens is None:
        cu_seqlens = torch.arange(0, sequences + 1, device=q.device, dtype=torch.int32)
        num_accepted = torch.ones(sequences, device=q.device, dtype=torch.int32)
    accepted = num_accepted.to(dtype=torch.int32)
    lengths = _seq_lengths(cu_seqlens)

    query, key, value = _l2_qkv(q, k, v)
    gate = -torch.exp(A_log) * F.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float()).to(dtype=torch.bfloat16)

    accepted_pos = cu_seqlens[:-1].to(dtype=torch.long) + accepted.to(dtype=torch.long) - 1
    gathered = _gather_state(state, read_indices)
    compact = torch.empty(
        total,
        value_heads,
        value_dim,
        key_dim,
        dtype=torch.bfloat16,
        device=state.device,
    )
    compact.index_copy_(0, accepted_pos, gathered)
    output = torch_npu.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        compact,
        beta=beta.contiguous(),
        scale=key_dim**-0.5,
        actual_seq_lengths=lengths.contiguous(),
        ssm_state_indices=torch.arange(total, device=state.device, dtype=torch.int32),
        num_accepted_tokens=accepted.contiguous(),
        g=gate.contiguous(),
    )
    _scatter_state(state, write_indices, compact)
    return output.unsqueeze(0)
