import torch
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Gemma3PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.embed_scale = torch.tensor(
            network_config["hidden_size"] ** 0.5,
            dtype=torch.float32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        return

    def _image_embed_scale_mask(self, input_ids, infer_state):
        # Text embeddings are scaled by sqrt(hidden). Vision features replace the
        # image-span rows unscaled. Prompt cache and chunked prefill can drop
        # <start_of_image> from this chunk, so the span has to come from the
        # image metadata rather than a scan of the current input ids.
        weight_mask = torch.empty((input_ids.shape[0],), dtype=torch.float32, device=input_ids.device)
        weight_mask[:] = self.embed_scale
        ready_cache_len = infer_state.b_ready_cache_len
        q_start_loc = infer_state.b_q_start_loc
        q_seq_len = infer_state.b_q_seq_len
        for batch_id, params in enumerate(infer_state.multimodal_params):
            cache_len = int(ready_cache_len[batch_id])
            flat_start = int(q_start_loc[batch_id])
            new_len = int(q_seq_len[batch_id])
            chunk_end = cache_len + new_len
            for img in params["images"]:
                image_start = img["start_idx"]
                image_end = image_start + img["token_num"]
                overlap_start = max(image_start, cache_len)
                overlap_end = min(image_end, chunk_end)
                if overlap_start < overlap_end:
                    begin = flat_start + overlap_start - cache_len
                    end = flat_start + overlap_end - cache_len
                    weight_mask[begin:end] = 1.0
        return weight_mask

    def context_forward(self, input_ids, infer_state, layer_weight):
        img_start_token_ids = []
        img_token_lens = []
        img_start_locs_in_cache = []
        device = layer_weight.wte_weight_.weight.device
        dtype = layer_weight.wte_weight_.weight.dtype
        hidden_size = layer_weight.wte_weight_.weight.shape[1]
        weight_mask = self._image_embed_scale_mask(input_ids, infer_state)

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"]:
                # skip the same image
                if img["token_id"] in img_start_token_ids:
                    continue
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs_in_cache.append(img["start_index_in_embed_cache"])
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        cpu_embed_cache_client = g_infer_context.cpu_embed_cache_client
        cpu_embed_cache_tensor = (
            torch.empty((0, 0, hidden_size), dtype=dtype, device=device)
            if cpu_embed_cache_client is None
            else cpu_embed_cache_client.cpu_embed_cache_tensor
        )
        assert cpu_embed_cache_tensor.shape[2] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image embed dimension is {cpu_embed_cache_tensor.shape[2]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        img_start_token_ids = torch.tensor(img_start_token_ids, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_token_lens = torch.tensor(img_token_lens, dtype=torch.long, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        img_start_locs_in_cache = torch.tensor(
            img_start_locs_in_cache, dtype=torch.long, device="cpu", pin_memory=True
        ).cuda(non_blocking=True)

        multimodal_emb(
            out=out,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_.weight,
            embed_cache=cpu_embed_cache_tensor,
            img_token_lens=img_token_lens,
            img_start_token_ids=img_start_token_ids,
            img_start_locs_in_cache=img_start_locs_in_cache,
            tp_text_start_token_id=layer_weight.wte_weight_.tp_vocab_start_id,
            tp_text_end_token_id=layer_weight.wte_weight_.tp_vocab_end_id,
            tp_world_size=self.tp_world_size_,
        )
        input_dtype = out.dtype
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (out.float() * weight_mask.unsqueeze(1).float()).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale).to(input_dtype)
