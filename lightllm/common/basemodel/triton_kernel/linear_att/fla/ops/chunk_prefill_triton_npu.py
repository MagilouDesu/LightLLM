import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import extract_slice, insert_slice

from .index import prepare_chunk_indices, prepare_chunk_offsets


_CHUNK_SIZE = 64
_SUPPORTED_DIMS = (64, 128)


def _cube_count() -> int:
    return triton.runtime.driver.active.utils.get_aicore_num()


def _vector_count() -> int:
    return triton.runtime.driver.active.utils.get_aivector_core_num()


def _packed_heads(x: torch.Tensor) -> torch.Tensor:
    # conv 把 token 收成连续的，dim 步长不再是 1。kernel 按 head * dim、列步长 1 来读。
    if x.stride(-1) != 1 or x.stride(-2) != x.shape[-1]:
        return x.contiguous()
    return x


@triton.jit
def _chunk_key_tiles(
    q,
    k,
    q_tiles,        # [num_chunks, key_heads, _CHUNK_SIZE, key_dim]
    k_tiles,        # [num_chunks, key_heads, _CHUNK_SIZE, key_dim]
    gram_tiles,     # [num_chunks, key_heads, _CHUNK_SIZE, _CHUNK_SIZE]
    score_tiles,    # [num_chunks, key_heads, _CHUNK_SIZE, _CHUNK_SIZE]
    cu_seqlens,
    chunk_indices,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    K_HEADS: tl.constexpr,
    K_DIM: tl.constexpr,
    BT: tl.constexpr,
    BLOCKS,
):
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        # 每个 program 处理一个 (chunk, k_head)
        chunk = block // K_HEADS
        key_head = block % K_HEADS

        sequence = tl.load(chunk_indices + 2 * chunk).to(tl.int32)
        local_chunk = tl.load(chunk_indices + 2 * chunk + 1).to(tl.int32)
        begin = tl.load(cu_seqlens + sequence)
        end = tl.load(cu_seqlens + sequence + 1)

        length = end - begin
        row = tl.arange(0, BT)
        query = tl.load(
            tl.make_block_ptr(
                q + begin * STRIDE_Q + key_head * K_DIM,
                (length, K_DIM),
                (STRIDE_Q, 1),
                (local_chunk * BT, 0),
                (BT, K_DIM),
                (1, 0),
            ),
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)
        key = tl.load(
            tl.make_block_ptr(
                k + begin * STRIDE_K + key_head * K_DIM,
                (length, K_DIM),
                (STRIDE_K, 1),
                (local_chunk * BT, 0),
                (BT, K_DIM),
                (1, 0),
            ),
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)

        # 归一化 q 和 k。越界行已由 boundary_check 填 0，归一化后仍是 0
        query = query / tl.sqrt(tl.sum(query * query, axis=1) + 1e-6)[:, None]
        key = key / tl.sqrt(tl.sum(key * key, axis=1) + 1e-6)[:, None]

        # 计算 q @ k 和 k @ k^T
        key_gram = tl.dot(key, tl.trans(key), input_precision="ieee")
        query_scores = tl.dot(query, tl.trans(key), input_precision="ieee")

        columns = tl.arange(0, K_DIM)
        vector_base = ((chunk * K_HEADS + key_head) * BT) * K_DIM
        pair_base = ((chunk * K_HEADS + key_head) * BT) * BT
        tl.store(q_tiles + vector_base + row[:, None] * K_DIM + columns[None, :], query.to(tl.bfloat16))
        tl.store(k_tiles + vector_base + row[:, None] * K_DIM + columns[None, :], key.to(tl.bfloat16))
        tl.store(gram_tiles + pair_base + row[:, None] * BT + row[None, :], key_gram)
        tl.store(score_tiles + pair_base + row[:, None] * BT + row[None, :], query_scores)


@triton.jit
def _chunk_value_solve(
    v,
    g,
    beta,
    k_tiles,
    gram_tiles,
    score_tiles,
    gate_cumsum,
    attention,
    inverse,
    scaled_values,
    scaled_keys,
    cu_seqlens,
    chunk_indices,
    STRIDE_V: tl.constexpr,
    STRIDE_G: tl.constexpr,
    STRIDE_BETA: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BLOCKS,
):
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        # 每个 program 处理一个 (chunk, value head)
        chunk = block // H
        head = block % H

        sequence = tl.load(chunk_indices + 2 * chunk).to(tl.int32)
        local_chunk = tl.load(chunk_indices + 2 * chunk + 1).to(tl.int32)
        begin = tl.load(cu_seqlens + sequence)
        end = tl.load(cu_seqlens + sequence + 1)

        length = end - begin
        row = tl.arange(0, BT)
        token = begin + local_chunk * BT + row
        valid = token < end
        key_head = head // (H // HG)

        # 读出这个 value head 的 log(alpha) 和 beta
        gate = tl.load(g + token * STRIDE_G + head, mask=valid, other=0).to(tl.float32)
        update_scale = tl.load(beta + token * STRIDE_BETA + head, mask=valid, other=0).to(tl.float32)
        cumulative = tl.cumsum(gate)
        tl.store(gate_cumsum + (chunk * H + head) * BT + row, cumulative)
        decay = tl.exp(tl.minimum(cumulative[:, None] - cumulative[None, :], 0))

        # 用 k_i^T k_j、beta 和 alpha 衰减求 T^{-1}
        pair_base = ((chunk * HG + key_head) * BT) * BT
        key_gram = tl.load(gram_tiles + pair_base + row[:, None] * BT + row[None, :])
        matrix = -tl.where(row[:, None] > row[None, :], key_gram * decay * update_scale[:, None], 0)
        rows_in_chunk = end - (begin + local_chunk * BT)
        for index in range(2, tl.minimum(BT, rows_in_chunk)):
            coefficients = tl.reshape(extract_slice(matrix, [index, 0], [1, BT], [1, 1]), (BT,))
            coefficients += tl.sum(coefficients[:, None] * matrix, axis=0)
            matrix = insert_slice(matrix, coefficients[None, :], [index, 0], [1, BT], [1, 1])
        matrix += (row[:, None] == row[None, :]).to(tl.float32)
        value_pair = ((chunk * H + head) * BT) * BT
        tl.store(inverse + value_pair + row[:, None] * BT + row[None, :], matrix.to(tl.bfloat16))

        # q_i^T k_j 乘 alpha 衰减，只留 i >= j
        query_scores = tl.load(score_tiles + pair_base + row[:, None] * BT + row[None, :])
        scores = tl.where(row[:, None] >= row[None, :], query_scores * decay, 0)
        tl.store(attention + value_pair + row[:, None] * BT + row[None, :], scores.to(tl.bfloat16))

        # beta * v
        value = tl.load(
            tl.make_block_ptr(
                v + begin * STRIDE_V + head * DV,
                (length, DV),
                (STRIDE_V, 1),
                (local_chunk * BT, 0),
                (BT, DV),
                (1, 0),
            ),
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)
        value_columns = tl.arange(0, DV)
        value_base = ((chunk * H + head) * BT) * DV
        tl.store(
            scaled_values + value_base + row[:, None] * DV + value_columns[None, :],
            (value * update_scale[:, None]).to(tl.bfloat16),
        )

        # beta * (alpha_1 ... alpha_i) * k
        key_columns = tl.arange(0, DK)
        key_base = ((chunk * HG + key_head) * BT) * DK
        key = tl.load(k_tiles + key_base + row[:, None] * DK + key_columns[None, :]).to(tl.float32)
        scaled_base = ((chunk * H + head) * BT) * DK
        tl.store(
            scaled_keys + scaled_base + row[:, None] * DK + key_columns[None, :],
            (key * update_scale[:, None] * tl.exp(cumulative)[:, None]).to(tl.bfloat16),
        )


@triton.jit
def _chunk_value_wu(
    inverse,
    scaled_values,
    scaled_keys,
    values,
    weights,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BLOCKS,
):
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        # 每个 program 处理一个 (chunk, value head)
        chunk = block // H
        head = block % H

        # T^{-1} @ (beta * v)
        row = tl.arange(0, BT)
        pair_base = ((chunk * H + head) * BT) * BT
        matrix = tl.load(inverse + pair_base + row[:, None] * BT + row[None, :])
        value_columns = tl.arange(0, DV)
        value_base = ((chunk * H + head) * BT) * DV
        value = tl.load(scaled_values + value_base + row[:, None] * DV + value_columns[None, :])
        updated = tl.dot(matrix, value)
        tl.store(values + value_base + row[:, None] * DV + value_columns[None, :], updated)

        # T^{-1} @ (beta * (alpha_1 ... alpha_i) * k)
        key_columns = tl.arange(0, DK)
        key_base = ((chunk * H + head) * BT) * DK
        key = tl.load(scaled_keys + key_base + row[:, None] * DK + key_columns[None, :])
        weight = tl.dot(matrix, key)
        tl.store(weights + key_base + row[:, None] * DK + key_columns[None, :], weight)


@triton.jit
def _chunk_output(
    q_tiles,
    k_tiles,
    values,
    weights,
    gate_cumsum,
    attention,
    initial_state,
    output,
    cu_seqlens,
    state_indices,
    chunk_offsets,
    sequence_heads,
    STRIDE_O: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BT: tl.constexpr,
    SCALE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
):
    tiles = V // BLOCK_V
    for task in range(tl.program_id(0), sequence_heads * tiles, tl.num_programs(0)):
        # 每个 program 处理一条序列的一个 value head，以及 value 维上的一块
        sequence_head = task // tiles
        sequence = sequence_head // H
        head = sequence_head % H
        key_head = head // (H // HG)

        chunk_begin = tl.load(chunk_offsets + sequence)
        begin = tl.load(cu_seqlens + sequence)
        end = tl.load(cu_seqlens + sequence + 1)
        state_index = tl.load(state_indices + sequence)

        # 读出这个 value head 的初始状态，只取当前这块 value 维
        key_offsets = tl.arange(0, K)
        value_offsets = (task % tiles) * BLOCK_V + tl.arange(0, BLOCK_V)
        row = tl.arange(0, BT)
        state_ptr = initial_state + state_index * STATE_STRIDE + head * K * V
        state = tl.load(state_ptr + key_offsets[:, None] * V + value_offsets[None, :]).to(tl.float32)
        for chunk in range(tl.cdiv(end - begin, BT)):
            token = begin + chunk * BT + row
            valid = token < end
            chunk_id = chunk_begin + chunk

            # LV，以及用来从状态里减掉旧投影的 LU
            weight = tl.load(
                weights + ((chunk_id * H + head) * BT + row)[:, None] * K + key_offsets[None, :],
                valid[:, None],
                other=0,
            )
            value = tl.load(
                values + ((chunk_id * H + head) * BT + row)[:, None] * V + value_offsets[None, :],
                valid[:, None],
                other=0,
            )
            # LV - L(U^T S)
            state_bf16 = state.to(tl.bfloat16)
            updated = value.to(tl.float32) - tl.dot(weight, state_bf16).to(tl.float32)

            # 用 q 读当前状态，用块内注意力乘更新后的 value
            query = tl.load(
                q_tiles + ((chunk_id * HG + key_head) * BT + row)[:, None] * K + key_offsets[None, :],
                valid[:, None],
                other=0,
            )
            scores = tl.load(
                attention + ((chunk_id * H + head) * BT + row)[:, None] * BT + row[None, :],
                valid[:, None],
                other=0,
            )
            cumulative = tl.load(gate_cumsum + (chunk_id * H + head) * BT + row)
            updated_bf16 = updated.to(tl.bfloat16)
            result = tl.dot(query, state_bf16).to(tl.float32) * tl.exp(cumulative)[:, None] + tl.dot(
                scores, updated_bf16
            ).to(tl.float32)
            tl.store(
                output + token[:, None] * STRIDE_O + head * V + value_offsets[None, :],
                (result * SCALE).to(output.dtype.element_ty),
                valid[:, None],
            )

            # S = (alpha_1 ... alpha_C) * S + K^T @ (updated * alpha_{i+1} ... alpha_C)
            valid_count = tl.minimum(BT, end - begin - chunk * BT)
            last_gate = tl.load(gate_cumsum + (chunk_id * H + head) * BT + valid_count - 1)
            updated = (updated * tl.exp(last_gate - cumulative)[:, None]).to(tl.bfloat16)
            key = tl.trans(
                tl.load(
                    k_tiles + ((chunk_id * HG + key_head) * BT + row)[:, None] * K + key_offsets[None, :],
                    valid[:, None],
                    other=0,
                )
            )
            state = state * tl.exp(last_gate) + tl.dot(key, updated).to(tl.float32)
        # 写回这条序列的最终状态
        tl.store(
            state_ptr + key_offsets[:, None] * V + value_offsets[None, :],
            state.to(initial_state.dtype.element_ty),
        )


def chunk_prefill_triton_npu(
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
    q = _packed_heads(q)
    k = _packed_heads(k)
    v = _packed_heads(v)
    # q、k、v 从一个 token 到下一个 token 的步长
    stride_q = q.stride(1)
    stride_k = k.stride(1)
    stride_v = v.stride(1)
    k_heads, k_dim = k.shape[2:]
    value_heads, value_dim = v.shape[2:]

    if k_dim not in _SUPPORTED_DIMS or value_dim not in _SUPPORTED_DIMS:
        raise ValueError(f"head dims must be {_SUPPORTED_DIMS}, got Dk={k_dim} Dv={value_dim}")

    if scale is None:
        scale = k_dim**-0.5

    # v 是混合 QKV 里切出来的，token 之间有空隙。按形状分配，输出才是连续的
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    # 把每条序列切成 64 个 token 一块，形状分别是 [num_chunks, 2] 和 [序列数 + 1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, _CHUNK_SIZE)
    num_chunks = len(chunk_indices)

    # 按 chunk 归一化 q 和 k，[1, num_tokens, k_heads, k_dim] -> [num_chunks, k_heads, _CHUNK_SIZE, k_dim]
    q_tiles = torch.empty(
        num_chunks, k_heads, _CHUNK_SIZE, k_dim,
        device=q.device,
        dtype=torch.bfloat16,
    )
    k_tiles = torch.empty_like(q_tiles)

    # k_i @ k_j 和 q_i @ k_j 的计算仅依赖归一化的 q 和 k，可以提前计算
    gram_tiles = torch.empty(
        num_chunks, k_heads, _CHUNK_SIZE, _CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    score_tiles = torch.empty_like(gram_tiles)

    key_blocks = num_chunks * k_heads
    _chunk_key_tiles[(min(key_blocks, _cube_count()),)](
        q,
        k,
        q_tiles,
        k_tiles,
        gram_tiles,
        score_tiles,
        cu_seqlens,
        chunk_indices,
        stride_q,
        stride_k,
        k_heads,
        k_dim,
        _CHUNK_SIZE,
        key_blocks,
        multibuffer=False,
        unit_flag=True,
    )

    # 块内 log(alpha) 的前缀和，对应 alpha 连乘
    gate_cumsum = torch.empty(
        num_chunks, value_heads, _CHUNK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    # q_i^T k_j 乘上 token 之间的 alpha 衰减，只留 i >= j
    attention = torch.empty(
        num_chunks, value_heads, _CHUNK_SIZE, _CHUNK_SIZE,
        device=q.device,
        dtype=torch.bfloat16,
    )

    # T^{-1}
    inverse = torch.empty_like(attention)
    # 计算 T^-1 @ (beta * v)，用于计算公式中的 LV
    values = torch.empty(
        num_chunks, value_heads, _CHUNK_SIZE, value_dim,
        device=q.device,
        dtype=torch.bfloat16,
    )
    # beta * v
    scaled_values = torch.empty_like(values)
    # beta * (alpha_1 ... alpha_i) * k，用于计算 L(US_0)
    scaled_keys = torch.empty(
        num_chunks, value_heads, _CHUNK_SIZE, k_dim,
        device=q.device,
        dtype=torch.bfloat16,
    )

    value_blocks = num_chunks * value_heads
    _chunk_value_solve[(min(value_blocks, _vector_count()),)](
        v,
        g,
        beta,
        k_tiles,
        gram_tiles,
        score_tiles,
        gate_cumsum,
        attention,
        inverse,
        scaled_values,
        scaled_keys,
        cu_seqlens,
        chunk_indices,
        stride_v,
        g.stride(1),
        beta.stride(1),
        value_heads,
        k_heads,
        k_dim,
        value_dim,
        _CHUNK_SIZE,
        value_blocks,
    )
    del gram_tiles, score_tiles

    # T^{-1} @ (beta * (alpha_1 ... alpha_i) * k)
    weights = torch.empty(
        num_chunks, value_heads, _CHUNK_SIZE, k_dim,
        device=q.device,
        dtype=torch.bfloat16,
    )
    # 计算 T^{-1} @ (beta * v) 和 weights
    _chunk_value_wu[(min(value_blocks, _cube_count()),)](
        inverse,
        scaled_values,
        scaled_keys,
        values,
        weights,
        value_heads,
        k_dim,
        value_dim,
        _CHUNK_SIZE,
        value_blocks,
        multibuffer=False,
        unit_flag=True,
    )
    del inverse, scaled_values, scaled_keys

    # 按序列扫描各个 chunk，写出输出并更新状态
    sequence_heads = (len(cu_seqlens) - 1) * value_heads
    # 序列头数少于 Cube 核时，把 value 维切成 64，凑满任务
    block_v = value_dim
    if sequence_heads < _cube_count():
        block_v = 64

    scan_tasks = sequence_heads * (value_dim // block_v)
    _chunk_output[(min(scan_tasks, _cube_count()),)](
        q_tiles,
        k_tiles,
        values,
        weights,
        gate_cumsum,
        attention,
        initial_state,
        output,
        cu_seqlens,
        state_indices,
        chunk_offsets,
        sequence_heads,
        output.stride(1),
        value_heads,
        k_heads,
        k_dim,
        value_dim,
        block_v,
        _CHUNK_SIZE,
        scale,
        initial_state.stride(0),
        multibuffer=False,
        unit_flag=True,
    )
    return output
