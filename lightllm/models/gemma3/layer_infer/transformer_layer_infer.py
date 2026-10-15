import torch
import torch.nn as nn
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.gemma3.layer_weights.transformer_layer_weight import Gemma3TransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm


class Gemma3TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = max(network_config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.eps_ = 1e-6
        self.head_dim_ = 256
        self.sliding_window_pattern = 6
        self.sliding_window_ = int(network_config.get("sliding_window", 1024) or 1024)
        return

    def _is_sliding_layer(self):
        return bool((self.layer_num_ + 1) % self.sliding_window_pattern)

    def _get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight
    ) -> torch.Tensor:
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        q = layer_weight.q_proj.mm(input)
        # kv = layer_weight.kv_proj.mm(input)
        # kv = kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        k = layer_weight.k_proj.mm(input)
        v = layer_weight.v_proj.mm(input)
        cache_kv = torch.cat(
            [k.view(-1, self.tp_k_head_num_, self.head_dim_), v.view(-1, self.tp_v_head_num_, self.head_dim_)], dim=1
        )

        # gemma3 use qk norm
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        k = cache_kv[:, 0 : self.tp_k_head_num_, :]

        q = layer_weight.q_norm_weight_(input=q.float(), eps=self.eps_, alloc_func=self.alloc_tensor).to(cache_kv.dtype)

        cache_kv[:, 0 : self.tp_k_head_num_, :] = layer_weight.k_norm_weight_(
            input=k.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(cache_kv.dtype)

        if self._is_sliding_layer():
            rotary_emb_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                cache_kv[:, 0 : self.tp_k_head_num_, :],
                infer_state.position_cos_local.to(q.dtype),
                infer_state.position_sin_local.to(q.dtype),
            )
        else:
            rotary_emb_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                cache_kv[:, 0 : self.tp_k_head_num_, :],
                infer_state.position_cos_global.to(q.dtype),
                infer_state.position_sin_global.to(q.dtype),
            )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        gate = layer_weight.gate_proj.mm(input)
        up = layer_weight.up_proj.mm(input)
        # gelu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn1_out = nn.functional.gelu(gate, approximate="tanh") * up
        input = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        ffn2_out = self._tpsp_reduce(input=ffn2_out, infer_state=infer_state)
        return ffn2_out

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight
    ) -> torch.Tensor:
        if not infer_state.has_image_tokens:
            return super()._context_attention_kernel(q, kv, infer_state, layer_weight)

        # Same-image tokens are bidirectional. Text tokens stay causal.
        # Sliding layers keep the window; full-attention layers do not.
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = self.alloc_tensor(_q.shape, q.dtype)
        if self._is_sliding_layer():
            sliding_window = (self.sliding_window_ - 1, 0)
        else:
            sliding_window = (-1, -1)
        context_attention_fwd_gemma4_mm(
            _q,
            _k,
            _v,
            o_tensor,
            infer_state.b_req_idx,
            infer_state.b_q_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_q_seq_len,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_image_token_end,
            sliding_window=sliding_window,
        )
        return o_tensor.view(q.shape)

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma3TransformerLayerWeight):
        input_embdings = input_embdings.to(torch.bfloat16)
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight).to(
            torch.bfloat16
        )
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = layer_weight.pre_feedforward_layernorm_weight_(
            input=input_embdings.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)

        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None

        ffn_out = layer_weight.post_feedforward_layernorm_weight_(
            input=ffn_out.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)

        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma3TransformerLayerWeight):
        input_embdings = input_embdings.to(torch.bfloat16)
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight).to(
            torch.bfloat16
        )
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = layer_weight.pre_feedforward_layernorm_weight_(
            input=input_embdings.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)

        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None

        ffn_out = layer_weight.post_feedforward_layernorm_weight_(
            input=ffn_out.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)

        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
