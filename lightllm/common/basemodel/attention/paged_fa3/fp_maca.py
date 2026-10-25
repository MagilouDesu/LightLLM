import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl

from lightllm.platform.base.attention import register_att_backend
from lightllm.utils.envs_utils import get_page_size

from ..base_att import AttControl
from .fp import PagedFa3AttBackend, PagedFa3DecodeAttState, PagedFa3PrefillAttState

try:
    from flash_attn import flash_attn_varlen_func as maca_flash_attn_varlen_func
except ImportError:
    maca_flash_attn_varlen_func = None

try:
    from flash_attn_3.flash_attn_3_interface import _flash_attn_forward as maca_fa3_forward
    from flash_attn_3.flash_attn_3_interface import flash_attn_with_kvcache as maca_fa3_with_kvcache
except ImportError:
    maca_fa3_forward = None
    maca_fa3_with_kvcache = None


def _maca_fa3_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if maca_fa3_forward is None or maca_fa3_with_kvcache is None:
        raise RuntimeError("MetaX paged attention requires the flash_attn_3 package")

    k_cache = k.view(-1, page_size, k.shape[-2], k.shape[-1])
    v_cache = v.view(-1, page_size, v.shape[-2], v.shape[-1])
    if q.dim() == 4:
        return maca_fa3_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
            num_splits=0,
        )
    out, *_ = maca_fa3_forward(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=0.0,
        num_splits=0,
    )
    return out


@triton.jit
def _gather_block_kv_kernel(
    K, V, OUT_K, OUT_V, REQ_TO_TOKEN, REQ_IDS, SEQ_LENS, CU_K,
    REQ_STRIDE: tl.constexpr, K_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr, V_HEAD_STRIDE: tl.constexpr,
    K_DIM_STRIDE: tl.constexpr, V_DIM_STRIDE: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (HEADS * DIM)
    feature = offsets % (HEADS * DIM)
    seq_len = tl.load(SEQ_LENS + batch)
    req = tl.load(REQ_IDS + batch)
    start = tl.load(CU_K + batch)
    valid = token < seq_len
    slot = tl.load(REQ_TO_TOKEN + req * REQ_STRIDE + token, mask=valid, other=0)
    kval = tl.load(K + slot * K_STRIDE + feature // DIM * K_HEAD_STRIDE
                   + feature % DIM * K_DIM_STRIDE, mask=valid, other=0)
    vval = tl.load(V + slot * V_STRIDE + feature // DIM * V_HEAD_STRIDE
                   + feature % DIM * V_DIM_STRIDE, mask=valid, other=0)
    output_offset = (start + token) * HEADS * DIM + feature
    tl.store(OUT_K + output_offset, kval, mask=valid)
    tl.store(OUT_V + output_offset, vval, mask=valid)


@register_att_backend(name="paged_fa3", category="standard", platforms=("maca",), validate_name="fa3")
class PagedFa3MacaAttBackend(PagedFa3AttBackend):

    def __init__(self, model, page_size=None):
        super().__init__(model=model, page_size=page_size or get_page_size())
        if self.page_size % 16 != 0:
            raise ValueError(
                "MetaX FlashAttention requires PAGE_SIZE to be a multiple "
                f"of 16, but got PAGE_SIZE={self.page_size}"
            )
        if maca_fa3_forward is None or maca_fa3_with_kvcache is None:
            raise RuntimeError("MetaX paged FlashAttention requires flash_attn_3")

    def create_att_prefill_state(self, infer_state):
        return PagedFa3MacaPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedFa3MacaDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3MacaPrefillAttState(PagedFa3PrefillAttState):

    def _normal_prefill_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)
        if att_control.use_att_sink:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")
        return _maca_fa3_paged(
            q=q,
            k=k,
            v=v,
            page_size=self.backend.page_size,
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=1.0 / (q.shape[-1] ** 0.5),
            causal=True,
            window_size=window_size,
            cu_seqlens_q=self.cu_seqlens_q,
        )


@dataclasses.dataclass
class PagedFa3MacaDecodeAttState(PagedFa3DecodeAttState):

    supports_noncausal_block = True

    def _block_decode_att(self, q, k, v, window_size, sink_weight, alloc_func):
        if sink_weight is not None:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")
        req_to_token = self.backend.model.req_manager.req_to_token_indexs
        shape = (self.b_block_req_idx.shape[0] * self.block_max_kv_len, k.shape[1], k.shape[2])
        packed_k = alloc_func(shape, dtype=k.dtype, device=k.device)
        packed_v = alloc_func(shape, dtype=v.dtype, device=v.device)
        _gather_block_kv_kernel[
            (triton.cdiv(self.block_max_kv_len * k.shape[1] * k.shape[2], 256), self.b_block_req_idx.shape[0])
        ](
            k, v, packed_k, packed_v, req_to_token, self.b_block_req_idx,
            self.b_att_seq_len, self.cu_seqlens_k,
            req_to_token.stride(0), k.stride(0), v.stride(0),
            k.stride(1), v.stride(1), k.stride(2), v.stride(2),
            k.shape[1], k.shape[2], 256,
        )
        return maca_flash_attn_varlen_func(
            q=q, k=packed_k, v=packed_v,
            cu_seqlens_q=self.cu_seqlens_q, cu_seqlens_k=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len, max_seqlen_k=self.block_max_kv_len,
            softmax_scale=q.shape[-1] ** -0.5, causal=False,
            window_size=window_size, softcap=0.0,
        )

    def _normal_decode_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)
        sink_weight = att_control.sink_weight if att_control.use_att_sink else None

        if not self.causal:
            if q.shape[0] != self.infer_state.batch_size:
                raise ValueError("Unexpected GPU decode query shape for noncausal block")
            return self._block_decode_att(q, k, v, window_size, sink_weight, alloc_func)

        if sink_weight is not None:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        if self.has_partial_block:
            return _maca_fa3_paged(
                q=q,
                k=k,
                v=v,
                page_size=self.backend.page_size,
                page_table=self.page_table,
                cache_seqlens=self.b_att_seq_len,
                max_seqlen_q=self.decode_max_q_seq_len,
                softmax_scale=sm_scale,
                causal=self.causal,
                window_size=window_size,
                cu_seqlens_q=self.cu_seqlens_q,
            )

        att_batch_size = self.page_table.shape[0]
        expected_tokens = att_batch_size * self.decode_max_q_seq_len
        if q.shape[0] != expected_tokens:
            raise ValueError(
                "Unexpected MetaX decode query shape: "
                f"q tokens={q.shape[0]}, batch={att_batch_size}, "
                f"q_len={self.decode_max_q_seq_len}"
            )
        q_bshd = q.view(att_batch_size, self.decode_max_q_seq_len, q.shape[1], q.shape[2])
        output = _maca_fa3_paged(
            q=q_bshd,
            k=k,
            v=v,
            page_size=self.backend.page_size,
            page_table=self.page_table,
            cache_seqlens=self.b_att_seq_len,
            max_seqlen_q=self.decode_max_q_seq_len,
            softmax_scale=sm_scale,
            causal=self.causal,
            window_size=window_size,
        )
        return output.view_as(q)
