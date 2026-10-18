import dataclasses

import torch

from lightllm.platform import get_backend
from lightllm.platform.base.attention import register_att_backend

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from .mlu_decode_attention import (
    decode_block_table_width,
    mlu_decode_attention_fwd,
    prepare_mlu_decode_block_table,
)
from .mlu_flash_attention import mlu_context_attention_fwd, prepare_mlu_flash_pack


@register_att_backend(name="fa3", category="standard", platforms=("mlu",))
class MluFa3AttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "MluFa3PrefillAttState":
        return MluFa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "MluFa3DecodeAttState":
        return MluFa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class MluFa3PrefillAttState(BasePrefillAttState):
    mlu_pack: object = None

    def init_state(self):
        self.mlu_pack = None
        if get_backend().graph.is_capturing():
            return
        self.mlu_pack = prepare_mlu_flash_pack(self.infer_state)

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_alibi is False
        assert att_control.use_att_sink is False
        if att_control.use_sliding_window:
            sliding_window = att_control.sliding_window
        else:
            sliding_window = (-1, -1)

        out = alloc_func(q.shape, q.dtype, device=q.device)
        if self.mlu_pack is None:
            from ...triton_kernel.att.prefill_att.context_flashattention_nopad import context_attention_fwd

            context_attention_fwd(
                q,
                k,
                v,
                out,
                self.infer_state.b_req_idx,
                self.infer_state.b_q_start_loc,
                self.infer_state.b_seq_len,
                self.infer_state.b_ready_cache_len,
                self.infer_state.max_q_seq_len,
                self.infer_state.req_manager.req_to_token_indexs,
                sliding_window=sliding_window,
            )
            return out

        mlu_context_attention_fwd(q, k, v, out, self.mlu_pack, sliding_window=sliding_window)
        return out


@dataclasses.dataclass
class MluFa3DecodeAttState(BaseDecodeAttState):
    block_tables: torch.Tensor = None

    def init_state(self):
        width = decode_block_table_width(
            self.backend.model,
            int(self.infer_state.b_req_idx.shape[0]),
            int(self.infer_state.max_kv_seq_len),
        )
        self.block_tables = prepare_mlu_decode_block_table(self.infer_state, width)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_alibi is False
        assert att_control.use_att_sink is False
        if att_control.use_sliding_window:
            window_left = int(att_control.sliding_window[0])
        else:
            window_left = -1
        out = alloc_func(q.shape, q.dtype, device=q.device)
        mlu_decode_attention_fwd(
            q=q,
            k=k,
            v=v,
            out=out,
            block_tables=self.block_tables,
            context_lens=self.infer_state.b_seq_len,
            window_left=window_left,
        )
        return out
