import torch
import triton
import triton.language as tl


@triton.jit
def _vocab_tile_topk(
    logits,
    tile_vals,
    tile_idx,
    vocab,
    n_tokens,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    x = tl.load(logits + offs * n_tokens + token, mask=mask, other=-1.0e20).to(tl.float32)
    idx = tl.where(mask, offs, 2147483647)
    x = tl.where(mask, x, -1.0e20)
    slot = tl.arange(0, TOP_K)
    vals = tl.full([TOP_K], -1.0e20, tl.float32)
    ids = tl.full([TOP_K], 2147483647, tl.int32)
    for k in tl.static_range(TOP_K):
        peak = tl.max(x, axis=0)
        pos = tl.min(tl.where(x == peak, idx, 2147483647), axis=0)
        vals = tl.where(slot == k, peak, vals)
        ids = tl.where(slot == k, pos, ids)
        x = tl.where(idx == pos, -1.0e20, x)
    base = (token * tl.num_programs(1) + tile) * TOP_K
    tl.store(tile_vals + base + slot, vals)
    tl.store(tile_idx + base + slot, ids)


def vocab_tile_topk(logits_vt: torch.Tensor, k: int, alloc_func) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over dim 0 of a contiguous [vocab, token] bf16 matrix.

    Each program keeps 16 values from a 4096-row tile. A small torch.topk
    merges the tiles. This matches a full top-k and avoids a [token, vocab] copy.
    """

    vocab, n_tokens = logits_vt.shape
    if n_tokens == 0:
        empty_vals = logits_vt.new_empty((0, k), dtype=torch.float32)
        empty_ids = torch.empty((0, k), dtype=torch.int64, device=logits_vt.device)
        return empty_vals, empty_ids

    block = 4096
    ntiles = triton.cdiv(vocab, block)
    tile_vals = alloc_func((n_tokens, ntiles, k), dtype=torch.float32, device=logits_vt.device)
    tile_idx = alloc_func((n_tokens, ntiles, k), dtype=torch.int32, device=logits_vt.device)
    _vocab_tile_topk[(n_tokens, ntiles)](
        logits_vt,
        tile_vals,
        tile_idx,
        vocab,
        n_tokens,
        TOP_K=k,
        BLOCK=block,
        num_warps=4,
    )
    flat_vals = tile_vals.reshape(n_tokens, ntiles * k)
    flat_idx = tile_idx.reshape(n_tokens, ntiles * k)
    top_vals, top_pos = torch.topk(flat_vals, k=k, dim=-1)
    top_idx = torch.gather(flat_idx, 1, top_pos).to(torch.int64)
    return top_vals, top_idx
