from typing import Optional, Tuple

import torch
from .fused_gdn_gating_910b2c import fused_gdn_gating_910b2c


def _mlu_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = a.float() + dt_bias.float()
    scaled = beta * x
    softplus_x = torch.where(scaled <= threshold, (1.0 / beta) * torch.log1p(torch.exp(scaled)), x)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float()).to(dtype=b.dtype).float()
    return g, beta_output


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    run_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.device.type == "mlu" and a.shape[0] >= 1024 and run_config is None:
        return _mlu_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold)
    if (
        a.shape[0] >= 128
        and a.shape[1] == 12
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.stride(1) == b.stride(1) == 1
        and run_config is None
        and a.device.type == "npu"
        and torch.npu.get_device_name(a.device) == "Ascend910B4"
    ):
        from .fused_gdn_gating_910b4 import fused_gdn_gating_910b4

        return fused_gdn_gating_910b4(A_log, a, b, dt_bias, beta, threshold)

    return fused_gdn_gating_910b2c(A_log, a, b, dt_bias, beta, threshold, run_config=run_config)
