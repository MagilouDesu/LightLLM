import os

_IMPLS = ("triton", "torch_npu")


def get_npu_gdn_prefill_impl() -> str:
    impl = os.getenv("LIGHTLLM_NPU_GDN_PREFILL_IMPL", "triton")
    if impl not in _IMPLS:
        raise ValueError("LIGHTLLM_NPU_GDN_PREFILL_IMPL must be triton or torch_npu")
    return impl


def get_npu_gdn_decode_impl() -> str:
    impl = os.getenv("LIGHTLLM_NPU_GDN_DECODE_IMPL", "triton")
    if impl not in _IMPLS:
        raise ValueError("LIGHTLLM_NPU_GDN_DECODE_IMPL must be triton or torch_npu")
    return impl


def npu_gdn_recurrent(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    read_indices,
    write_indices,
    cu_seqlens=None,
    num_accepted=None,
):
    if get_npu_gdn_decode_impl() == "torch_npu":
        from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.torch_npu_recurrent import (
            torch_npu_recurrent
        )

        return torch_npu_recurrent(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            num_accepted=num_accepted,
        )

    # 非 MTP 模式
    if cu_seqlens is None:
        from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_npu import (
            fused_recurrent_decode_npu,
        )

        return fused_recurrent_decode_npu(
            q=q,
            k=k,
            v=v,
            initial_state=state,
            ssm_state_indices=read_indices,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a,
            b_raw=b,
        )

    from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
        mtp_fused_recurrent_gated_delta_rule,
    )

    output, _ = mtp_fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=state,
        final_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=read_indices,
        ssm_state_write_indices=write_indices,
        num_accepted_tokens=num_accepted,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a,
        b_raw=b,
    )
    return output
