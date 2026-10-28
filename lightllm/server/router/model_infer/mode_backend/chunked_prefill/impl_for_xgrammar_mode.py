import functools
from typing import List

import torch

from .impl import ChunkedPrefillBackend
from lightllm.common.basemodel.triton_kernel.post_process.apply_constraint_mask import (
    MASKED_LOGIT,
    apply_constraint_mask,
)
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.support_overlap = False
        self.prefill_mask_func = self._prefill_mask_callback
        self.decode_mask_func = self._decode_mask_callback
        self.extra_post_req_handle_func = self._update_xgrammar_fsm

    def init_custom(self):
        import xgrammar as xgr
        from transformers import PreTrainedTokenizerBase

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        hf_tokenizer = self.tokenizer
        while not isinstance(hf_tokenizer, PreTrainedTokenizerBase) and hasattr(hf_tokenizer, "tokenizer"):
            hf_tokenizer = hf_tokenizer.tokenizer
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=8)
        max_mask_rows = self.args.running_max_req_size * (self.args.mtp_step + 1)
        self._xgrammar_bitmask_cpu = xgr.allocate_token_bitmask(
            max_mask_rows, self.tokenizer_info.vocab_size
        ).pin_memory()
        self._xgrammar_enabled_cpu = torch.zeros(max_mask_rows, dtype=torch.bool, pin_memory=True)
        self._xgrammar_bitmask_gpu = None
        self._xgrammar_enabled_gpu = None

        @functools.lru_cache(maxsize=200)
        def get_cached_grammar(constraint_type: str, grammar: str):
            logger.info(f"grammar cache miss for {constraint_type}: '{grammar}'")
            try:
                if constraint_type == "grammar":
                    if grammar == "json":
                        return self.xgrammar_compiler.compile_builtin_json_grammar()
                    return self.xgrammar_compiler.compile_grammar(grammar)
                if constraint_type == "schema":
                    return self.xgrammar_compiler.compile_json_schema(grammar)
                raise ValueError(f"Unknown xgrammar type: {constraint_type}")
            except Exception as e:
                logger.error(f"Failed to compile {constraint_type}: {e}")
                raise

        self.get_cached_grammar = get_cached_grammar

    def _decode_mask_callback(
        self, run_reqs: List[InferReq], logits: torch.Tensor, b_mtp_index: torch.Tensor = None
    ):
        masked = False
        if self._has_output_constraints(run_reqs):
            self._init_req_xgrammar_matchers(run_reqs)
            masked = self._apply_xgrammar_masks(run_reqs, logits, b_mtp_index)
        if not masked:
            self._mask_padding_logits(logits)

    def _prefill_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        masked = False
        if self._has_output_constraints(run_reqs):
            self._init_req_xgrammar_matchers(run_reqs)
            masked = self._apply_xgrammar_masks(run_reqs, logits)
        if not masked:
            self._mask_padding_logits(logits)

    def _mask_padding_logits(self, logits: torch.Tensor) -> None:
        vocab_size = self.tokenizer_info.vocab_size
        if logits.shape[1] > vocab_size:
            logits[:, vocab_size:] = MASKED_LOGIT

    @staticmethod
    def _has_output_constraints(run_reqs: List[InferReq]) -> bool:
        return any(
            req.sampling_param.guided_grammar is not None or req.sampling_param.guided_json is not None
            for req in run_reqs
        )

    def _update_xgrammar_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        if not hasattr(req_obj.sampling_param, "xgrammar_matcher"):
            return

        matcher = req_obj.sampling_param.xgrammar_matcher
        assert matcher.accept_token(next_token_id)
        if matcher.is_terminated():
            req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)

    def _apply_xgrammar_masks(
        self, run_reqs: List[InferReq], logits: torch.Tensor, b_mtp_index: torch.Tensor = None
    ) -> bool:
        row_count = len(run_reqs)
        bitmask = self._xgrammar_bitmask_cpu[:row_count]
        bitmask_enabled = self._xgrammar_enabled_cpu[:row_count]
        bitmask_enabled.zero_()
        enabled_flags = bitmask_enabled.numpy()
        has_enabled_mask = False

        if b_mtp_index is None:
            for i, run_obj in enumerate(run_reqs):
                if run_obj.get_chuncked_input_token_len() != run_obj.get_cur_total_len():
                    continue
                sample_params = run_obj.sampling_param
                if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
                    sample_params.xgrammar_matcher.fill_next_token_bitmask(bitmask, i)
                    enabled_flags[i] = True
                    has_enabled_mask = True
        else:
            next_token_table = (
                g_infer_context.req_manager.req_sampling_params_manager.req_to_next_token_ids_cpu
            )
            group_start = 0
            while group_start < len(run_reqs):
                run_obj = run_reqs[group_start]
                group_end = group_start + 1
                while group_end < len(run_reqs) and run_reqs[group_end] is run_obj:
                    group_end += 1

                sample_params = run_obj.sampling_param
                if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
                    matcher = sample_params.xgrammar_matcher
                    group_size = group_end - group_start
                    draft_tokens = (
                        next_token_table[run_obj.req_idx, 1:group_size].tolist()
                        if group_size > 1
                        else []
                    )
                    accepted_draft_count = 0
                    try:
                        for offset in range(group_size):
                            if matcher.is_terminated():
                                break
                            matcher.fill_next_token_bitmask(bitmask, group_start + offset)
                            enabled_flags[group_start + offset] = True
                            has_enabled_mask = True
                            if offset < len(draft_tokens):
                                if not matcher.accept_token(draft_tokens[offset]):
                                    break
                                accepted_draft_count += 1
                    finally:
                        if accepted_draft_count:
                            matcher.rollback(accepted_draft_count)

                group_start = group_end

        if has_enabled_mask:
            device = logits.device
            if self._xgrammar_bitmask_gpu is None:
                self._xgrammar_bitmask_gpu = torch.empty_like(self._xgrammar_bitmask_cpu, device=device)
                self._xgrammar_enabled_gpu = torch.empty_like(self._xgrammar_enabled_cpu, device=device)
            bitmask_gpu = self._xgrammar_bitmask_gpu[:row_count]
            enabled_gpu = self._xgrammar_enabled_gpu[:row_count]
            # Disabled rows hold stale words; the kernel skips them via enabled_gpu.
            enabled_gpu.copy_(bitmask_enabled, non_blocking=True)
            bitmask_gpu.copy_(bitmask, non_blocking=True)
            apply_constraint_mask(
                logits=logits,
                req_to_bitmask=bitmask_gpu,
                req_to_bitmask_enabled=enabled_gpu,
                vocab_size=self.tokenizer_info.vocab_size,
            )
        return has_enabled_mask

    def _init_req_xgrammar_matchers(self, run_reqs: List[InferReq]):
        import xgrammar as xgr

        for run_obj in run_reqs:
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    ctx = self.get_cached_grammar("grammar", sample_params.guided_grammar)
                    sample_params.xgrammar_matcher = xgr.GrammarMatcher(ctx)
            elif sample_params.guided_json is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    ctx = self.get_cached_grammar("schema", sample_params.guided_json)
                    sample_params.xgrammar_matcher = xgr.GrammarMatcher(ctx)
