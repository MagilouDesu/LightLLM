import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.gemma4.triton_kernel.build_b_image_token_end import build_b_image_token_end


class Gemma3InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos_global = None
        self.position_sin_global = None
        self.position_sin_local = None
        self.position_cos_local = None
        # Per new token: image span end if the token is inside an image, else 0.
        # Image tokens attend bidirectionally inside that span; text stays causal.
        self.b_image_token_end = None
        self.has_image_tokens = False

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)

            self.position_cos_local = torch.index_select(model._cos_cached_local, 0, position_ids).view(
                position_ids.shape[0], -1
            )
            self.position_sin_local = torch.index_select(model._sin_cached_local, 0, position_ids).view(
                position_ids.shape[0], -1
            )

            self.position_cos_global = torch.index_select(model._cos_cached_global, 0, position_ids).view(
                position_ids.shape[0], -1
            )
            self.position_sin_global = torch.index_select(model._sin_cached_global, 0, position_ids).view(
                position_ids.shape[0], -1
            )
            self._build_b_image_token_end()
        else:
            position_ids = self.position_ids
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)

            self.position_cos_local = torch.index_select(model._cos_cached_local, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
            self.position_sin_local = torch.index_select(model._sin_cached_local, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )

            self.position_cos_global = torch.index_select(model._cos_cached_global, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
            self.position_sin_global = torch.index_select(model._sin_cached_global, 0, position_ids).view(
                self.b_seq_len.shape[0], -1
            )
        return

    def _build_b_image_token_end(self):
        device = self.position_ids.device
        self.b_image_token_end = torch.zeros(self.position_ids.shape[0], dtype=torch.int32, device=device)
        self.has_image_tokens = False
        if not self.multimodal_params:
            return

        b_image_start_idx = []
        b_image_len = []
        b_image_nums = []
        b_image_start_num = []
        image_start_num = 0
        for params in self.multimodal_params:
            b_image_start_num.append(image_start_num)
            images = params.get("images", [])
            b_image_nums.append(len(images))
            for img in images:
                b_image_start_idx.append(img["start_idx"])
                b_image_len.append(img["token_num"])
                image_start_num += 1

        if image_start_num == 0:
            return

        self.has_image_tokens = True
        build_b_image_token_end(
            b_image_start_idx=torch.tensor(b_image_start_idx, dtype=torch.int32).to(device=device, non_blocking=True),
            b_image_len=torch.tensor(b_image_len, dtype=torch.int32).to(device=device, non_blocking=True),
            b_image_nums=torch.tensor(b_image_nums, dtype=torch.int32).to(device=device, non_blocking=True),
            b_image_start_num=torch.tensor(b_image_start_num, dtype=torch.int32).to(device=device, non_blocking=True),
            b_q_start_loc=self.b_q_start_loc,
            b_ready_cache_len=self.b_ready_cache_len,
            b_q_seq_len=self.b_q_seq_len,
            b_image_token_end=self.b_image_token_end,
        )
