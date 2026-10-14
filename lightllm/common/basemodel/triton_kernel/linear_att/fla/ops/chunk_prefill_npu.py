import torch

from .npu_gdn_backend import get_npu_gdn_prefill_impl


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    denom = torch.sqrt(values.square().sum(dim=-1, keepdim=True) + 1e-6)
    return (values / denom).to(dtype=torch.bfloat16)


def _l2_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return _l2_normalize(q[0]), _l2_normalize(k[0]), v[0].to(dtype=torch.bfloat16)


def _with_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 2 else x


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x[0] if x.ndim == 3 else x


def _seq_lengths(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.int32)


def _gather_state(state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = state.index_select(0, indices.to(dtype=torch.long))
    return gathered.transpose(-1, -2).contiguous().to(dtype=torch.bfloat16)


def _scatter_state(state: torch.Tensor, indices: torch.Tensor, packed: torch.Tensor) -> None:
    written = packed.transpose(-1, -2).contiguous().to(dtype=state.dtype)
    state.index_copy_(0, indices.reshape(-1).to(dtype=torch.long), written)


def torch_npu_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    if get_npu_gdn_prefill_impl() == "triton":
        from .chunk_prefill_triton_npu import chunk_prefill_triton_npu

        return chunk_prefill_triton_npu(
            q,
            k,
            v,
            _with_batch(g),
            _with_batch(beta),
            initial_state,
            cu_seqlens,
            state_indices,
            scale,
        )

    import torch_npu

    assert q.shape[0] == 1 and k.shape[0] == 1 and v.shape[0] == 1
    _, _, key_head_num, key_dim = k.shape
    value_head_num, value_dim = v.shape[2], v.shape[3]
    assert value_head_num % key_head_num == 0
    assert 1 <= key_head_num <= 64 and 1 <= value_head_num <= 64
    assert 1 <= key_dim <= 128 and 1 <= value_dim <= 128
    sequence_num = cu_seqlens.shape[0] - 1
    assert state_indices.shape[0] == sequence_num
    if scale is None:
        scale = key_dim**-0.5

    query, key, value = _l2_qkv(q, k, v)
    gate = _rows(g).contiguous().to(dtype=torch.float32)
    update = _rows(beta).contiguous().to(dtype=torch.bfloat16)
    state = _gather_state(initial_state, state_indices)
    output, final_state = torch_npu.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta=update,
        initial_state=state,
        actual_seq_lengths=_seq_lengths(cu_seqlens),
        scale=float(scale),
        g=gate,
    )
    _scatter_state(initial_state, state_indices, final_state)
    return output.unsqueeze(0)
