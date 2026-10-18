import torch
from typing import Tuple

from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextTransformerLayerInfer,
)
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen35TransformerLayerInfer(Qwen3NextTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        # Initialize mrope section from config
        rope_scaling = network_config.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [11, 11, 10])
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device=self.target_device)
        self._mrope_h = int(mrope_section[1])
        self._mrope_w = int(mrope_section[2])

    def _mlu_mrope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        import torch_mlu_ops as tmo

        half = int(q.shape[-1] * self.partial_rotary_factor) // 2
        tokens = q.shape[0]
        masks = getattr(self, "_mrope_masks", None)
        if masks is None or masks[0].numel() != half or masks[0].device != q.device:
            offsets = torch.arange(half, device=q.device)
            h_mask = ((offsets % 3) == 1) & (offsets <= 3 * self._mrope_h)
            w_mask = ((offsets % 3) == 2) & (offsets <= 3 * self._mrope_w)
            masks = (~(h_mask | w_mask), h_mask, w_mask)
            self._mrope_masks = masks
        t_mask, h_mask, w_mask = masks
        cos_h = cos[..., :half]
        sin_h = sin[..., :half]
        mixed_cos = torch.where(t_mask, cos_h[0], torch.where(h_mask, cos_h[1], cos_h[2]))
        mixed_sin = torch.where(t_mask, sin_h[0], torch.where(h_mask, sin_h[1], sin_h[2]))
        buf = getattr(self, "_mrope_buf", None)
        if buf is None or buf[0].shape[0] < tokens or buf[0].device != q.device:
            cap = max(tokens, 64)
            buf = (
                torch.empty(cap, half * 2, dtype=q.dtype, device=q.device),
                torch.empty(cap, half * 2, dtype=q.dtype, device=q.device),
                torch.zeros(2, dtype=torch.int32, device=q.device),
            )
            self._mrope_buf = buf
        cos_buf, sin_buf, cu = buf
        cos_buf[:tokens, :half].copy_(mixed_cos)
        cos_buf[:tokens, half:].copy_(mixed_cos)
        sin_buf[:tokens, :half].copy_(mixed_sin)
        sin_buf[:tokens, half:].copy_(mixed_sin)
        cu[1] = tokens
        tmo.apply_rotary(q, sin_buf[:tokens], cos_buf[:tokens], None, cu, False, False, False, tokens)
        tmo.apply_rotary(k, sin_buf[:tokens], cos_buf[:tokens], None, cu, False, False, False, tokens)

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen35TransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        qkv_gate_out = layer_weight.qkvo_gate_proj.mm(input)
        qkv_out, o_gate = qkv_gate_out.split(
            [
                self.tp_q_head_num_ * self.head_dim_ + (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
                self.tp_q_head_num_ * self.head_dim_,
            ],
            dim=-1,
        )
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )

        infer_state.gate_logics_value = o_gate
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        q_heads = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        k_heads = cache_kv[:, : self.tp_k_head_num_, :]
        if q_heads.device.type == "mlu":
            self._mlu_mrope(q_heads, k_heads, infer_state.position_cos, infer_state.position_sin)
        else:
            mrope_triton_fused(
                q_heads,
                k_heads,
                infer_state.position_cos,
                infer_state.position_sin,
                self.mrope_section,
                is_interleaved=True,  # Qwen3 uses interleaved mrope
                partial_rotary_factor=self.partial_rotary_factor,
            )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv
