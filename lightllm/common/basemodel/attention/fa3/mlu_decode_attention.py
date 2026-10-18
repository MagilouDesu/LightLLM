import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def decode_block_table_width(model, batch_size: int, max_kv_seq_len: int) -> int:
    graph = getattr(model, "graph", None)
    if graph is not None and graph.can_run(batch_size=batch_size, max_len_in_batch=max_kv_seq_len):
        return int(graph.graph_max_len_in_batch)
    return max_kv_seq_len


def prepare_mlu_decode_block_table(infer_state, width: int) -> torch.Tensor:
    rows = infer_state.req_manager.req_to_token_indexs.index_select(
        0, infer_state.b_req_idx.to(dtype=torch.int64)
    )
    if rows.shape[1] >= width:
        return rows[:, :width].contiguous()
    table = torch.zeros((rows.shape[0], width), dtype=rows.dtype, device=rows.device)
    table[:, : rows.shape[1]] = rows
    return table


@torch.no_grad()
def mlu_decode_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    window_left: int = -1,
) -> torch.Tensor:
    logger.info("decode full attention on MLU uses torch_mlu_ops.single_query_cached_kv_attn")

    import torch_mlu_ops as tmo

    batch, q_heads, head_dim = q.shape
    q_in = q.view(batch, 1, q_heads, head_dim)
    out_in = out.view(batch, 1, q_heads, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    tmo.single_query_cached_kv_attn(
        q_in,
        k.unsqueeze(2),
        v.unsqueeze(2),
        out_in,
        block_tables,
        context_lens,
        None,
        None,
        None,
        int(block_tables.shape[1]),
        int(window_left),
        -1,
        scale,
        False,
        -1,
        None,
        None,
        q.dtype,
        torch.float32,
        -1,
        None,
        None,
        -1,
        None,
    )
    return out
