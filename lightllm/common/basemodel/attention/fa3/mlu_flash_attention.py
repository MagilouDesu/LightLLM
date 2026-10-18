from dataclasses import dataclass

import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_WORKSPACE = {}


@dataclass
class MluFlashPack:
    slots: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    k_buf: torch.Tensor
    v_buf: torch.Tensor
    max_q_seq_len: int
    max_kv_seq_len: int


def _workspace_buffers(total: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
    dev = torch.device(device)
    key = (dev.type, dev.index, dtype, kv_heads, head_dim)
    bufs = _WORKSPACE.get(key)
    cap = 0 if bufs is None else bufs[0].shape[0]
    if bufs is None or cap < total:
        cap = max(total, 2048)
        cap = 1 << (cap - 1).bit_length()
        k_buf = torch.empty((cap, kv_heads, head_dim), dtype=dtype, device=dev)
        v_buf = torch.empty((cap, kv_heads, head_dim), dtype=dtype, device=dev)
        _WORKSPACE[key] = (k_buf, v_buf)
    else:
        k_buf, v_buf = bufs
    return k_buf[:total], v_buf[:total]


def _pack_slots(
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_kv_seq_len: int,
) -> torch.Tensor:
    batch = b_req_idx.shape[0]
    if batch == 1:
        return req_to_token_indexs.index_select(0, b_req_idx.to(dtype=torch.int64)).view(-1)[:max_kv_seq_len]

    rows = req_to_token_indexs.index_select(0, b_req_idx.to(dtype=torch.int64))[:, :max_kv_seq_len]
    pos = torch.arange(max_kv_seq_len, device=rows.device)
    mask = pos.unsqueeze(0) < b_seq_len.unsqueeze(1)

    return rows[mask]


def prepare_mlu_flash_pack(infer_state) -> MluFlashPack:
    mem = infer_state.mem_manager
    max_kv = int(infer_state.max_kv_seq_len)
    slots = _pack_slots(
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        max_kv,
    )
    k_buf, v_buf = _workspace_buffers(slots.shape[0], mem.head_num, mem.head_dim, mem.dtype, slots.device)
    return MluFlashPack(
        slots=slots,
        cu_q=infer_state.b1_cu_q_seq_len,
        cu_kv=infer_state.b1_cu_kv_seq_len,
        k_buf=k_buf,
        v_buf=v_buf,
        max_q_seq_len=int(infer_state.max_q_seq_len),
        max_kv_seq_len=max_kv,
    )


@torch.no_grad()
def mlu_context_attention_fwd(
    q,
    k,
    v,
    o,
    pack: MluFlashPack,
    sliding_window=(-1, -1),
):
    logger.info("prefill full attention on MLU uses torch_mlu_ops.flash_attention")

    import torch_mlu_ops as tmo

    if sliding_window == (-1, -1):
        window_left, window_right = -1, -1
    else:
        window_left = int(sliding_window[0])
        window_right = int(sliding_window[1])

    torch.ops.aten.index_select.out(k, 0, pack.slots, out=pack.k_buf)
    torch.ops.aten.index_select.out(v, 0, pack.slots, out=pack.v_buf)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    tmo.flash_attention(
        q.contiguous(),
        pack.k_buf,
        pack.v_buf,
        o,
        pack.cu_q,
        pack.cu_kv,
        None,
        None,
        int(pack.max_q_seq_len),
        int(pack.max_kv_seq_len),
        scale,
        True,
        window_left,
        window_right,
        torch.float32,
        False,
    )
    return o
