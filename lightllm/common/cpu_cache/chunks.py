import torch
from typing import Iterator, List, Optional, Tuple


def iter_page_chunk_slices(
    chunks: List[torch.Tensor],
    page_indexes: torch.Tensor,
    mem_indexes: torch.Tensor,
    token_block: int,
    per_page: List[torch.Tensor],
) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """ 沐曦分段锁页后，按段把一次 CPU cache 搬运拆开。 """
    # 全局页号。mem_indexes 比整页 token 短时，缺的是页面前部，真正要搬的从 move_begin 对齐到 mem_indexes[0]。
    pages = page_indexes.detach().to(device="cpu", dtype=torch.int64).tolist()
    page_num = len(pages)
    token_num = int(mem_indexes.shape[0])
    move_begin = page_num * token_block - token_num

    # 每段缓存的全局页号起点。
    starts: List[int] = []
    page_total = 0
    for chunk in chunks:
        starts.append(page_total)
        page_total += int(chunk.shape[0])

    # 全局页号落到哪一段，以及段内下标。
    def locate(page_id: int) -> Tuple[int, int]:
        for chunk_id, start in enumerate(starts):
            end = starts[chunk_id + 1] if chunk_id + 1 < len(starts) else page_total
            if start <= page_id < end:
                return chunk_id, page_id - start
        raise RuntimeError(f"cpu cache page {page_id} is outside {page_total} pages")

    # 连续页合成一次搬运。未分配页（page_id < 0）挂在当前段上，段内下标记为 -1。
    # 开头还没遇到有效页时先占位，遇到第一段有效页再归过去；全是未分配页则归到第 0 段。
    runs: List[Tuple[Optional[int], List[Tuple[int, int]]]] = []
    for pos, page_id in enumerate(pages):
        if page_id < 0:
            if not runs:
                runs.append((None, []))
            runs[-1][1].append((pos, -1))
            continue
        chunk_id, local = locate(page_id)
        if not runs or (runs[-1][0] is not None and runs[-1][0] != chunk_id):
            runs.append((chunk_id, []))
        elif runs[-1][0] is None:
            runs[-1] = (chunk_id, runs[-1][1])
        runs[-1][1].append((pos, local))
    if runs and runs[0][0] is None:
        runs[0] = (0, runs[0][1])

    # 切出这一段的 token，以及和这些页对齐的附属张量（ready、big page id 等）。
    for chunk_id, items in runs:
        pos0 = items[0][0]
        pos1 = items[-1][0]
        run_begin = pos0 * token_block
        run_end = (pos1 + 1) * token_block
        overlap_begin = max(run_begin, move_begin)
        if overlap_begin >= run_end:
            continue
        mem_start = overlap_begin - move_begin
        mem_end = run_end - move_begin
        local_pages = torch.tensor([local for _, local in items], dtype=page_indexes.dtype, device=page_indexes.device)
        page_slices = [tensor[pos0 : pos1 + 1] for tensor in per_page]

        yield chunk_id, chunks[chunk_id], local_pages, mem_indexes[mem_start:mem_end], page_slices
