import triton
import triton.language as tl
import torch


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
    }
)
@triton.jit
def _mlu_gated_rmsnorm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    num_pid_m = tl.num_programs(0)
    W += group * N
    if HAS_BIAS:
        B += group * N

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    for row_start in range(pid_m * BLOCK_M, M, num_pid_m * BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        tile_mask = row_mask[:, None] & col_mask[None, :]

        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
        x = tl.load(x_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=tile_mask).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        rstd = 1 / tl.sqrt(var + eps)
        y = x * rstd[:, None] * w[None, :]
        if HAS_BIAS:
            y += b[None, :]
        if NORM_BEFORE_GATE:
            y *= z * tl.sigmoid(z)

        y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
        tl.store(y_ptrs, y, mask=tile_mask)


def mlu_gated_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
):
    m, n = x.shape
    if group_size is None:
        group_size = n
    assert n % group_size == 0
    assert z is not None and z.shape == (m, n)
    if out is None:
        out = torch.empty_like(x)

    block_m = 64
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(group_size))
    if group_size > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(block_n // 256, 1), 4)
    grid_m = min(triton.cdiv(m, block_m), 65535)
    _mlu_gated_rmsnorm_fwd_kernel[(grid_m, n // group_size)](
        x,
        out,
        weight,
        bias,
        z,
        x.stride(0),
        out.stride(0),
        z.stride(0),
        m,
        group_size,
        eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NORM_BEFORE_GATE=norm_before_gate,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
