import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        torch.cuda.init()
    from vllm import _custom_ops as ops

    vllm_ops = ops
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False
    vllm_ops = None
    logger.warning("vllm is not installed, you can't use the api of it. You can solve it by running `pip install vllm`.")

cutlass_scaled_mm = getattr(torch.ops._C, "cutlass_scaled_mm", None) if HAS_VLLM else None
