
"""
Copied from https://github.com/HazyResearch/flash-attention/blob/eff9fe6b8076df59d64d7a3f464696738a3c7c24/flash_attn/flash_attn_triton.py
update imports to use 'triton_pre_mlir'

*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math

import torch

import triton
import triton.language as tl

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX,
    pre_load_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(V_block_ptr.dtype.element_ty), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
):
    start_m = tl.program_id(0)
    # Select batch
    off_z = tl.program_id(1) // H
    # Select head within that batch
    off_h = tl.program_id(1) % H
    # Set Q block start offset. Q has shape [batch, seqlen, nheads, d]
    q_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_qm = offs_m * stride_qm
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q.dtype.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            4 - STAGE, offs_m, offs_n,
            N_CTX, pre_load_v,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            2, offs_m, offs_n,
            N_CTX, pre_load_v,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    m_ptrs = M + (off_z * H * N_CTX + off_h * N_CTX) + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    o_offset = off_z * stride_oz + off_h * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_om,
    stride_dob, stride_doh, stride_dom,
    nheads, seqlen_q, seqlen_q_rounded, headdim,
    BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
                 mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqkvz, stride_dqkvh, stride_dqkvn, stride_dqkvk,
    Z, H, N_CTX, IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    # Select batch
    off_z = tl.program_id(1) // H
    # Select head within that batch
    off_h = tl.program_id(1) % H
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    dqvk_offset = off_z * stride_dqkvz + off_h * stride_dqkvh
    q_offset = qvk_offset + start_m * BLOCK_M * stride_qm
    do_base_offset = off_z * stride_doz + off_h * stride_doh
    do_offset = do_base_offset + start_m * BLOCK_M * stride_dom
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dom, stride_dok),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_z * H * N_CTX + off_h * N_CTX
    l_ptrs = L + off_z * H * N_CTX + off_h * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(Q.dtype.element_ty)
    v = tl.load(V_block_ptr)
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM if causal is true.
    lo = start_m * BLOCK_M
    hi = N_CTX
    # loop over q, do
    for start_n in range(lo, hi, BLOCK_N):
        offs_m_curr = offs_n[:, None] + start_n
        # -- load q, do --
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # -- compute dv ----
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - Di
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_N, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_N, 0))
    # initialize pointers to output
    DK_block_ptr = tl.make_block_ptr(
        base=DK + dqvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dqkvn, stride_dqkvk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + dqvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dqkvn, stride_dqkvk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DV.dtype.element_ty))

@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqkvz, stride_dqkvh, stride_dqkvn, stride_dqkvk,
    Z, H, N_CTX, IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    # Select batch
    off_z = tl.program_id(1) // H
    # Select head within that batch
    off_h = tl.program_id(1) % H
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    dqvk_offset = off_z * stride_dqkvz + off_h * stride_dqkvh
    do_base_offset = off_z * stride_doz + off_h * stride_doh
    do_offset = do_base_offset + start_m * BLOCK_M * stride_dom
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dom, stride_dok),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_z * H * N_CTX + off_h * N_CTX
    l_ptrs = L + off_z * H * N_CTX + off_h * N_CTX
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q.dtype.element_ty)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, v)
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        dq += tl.dot(ds.to(Q.dtype.element_ty), tl.trans(k))
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + dqvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dqkvn, stride_dqkvk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ.dtype.element_ty))


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert bias is None, 'Bias for Triton FA not supported yet'
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q, dtype=v.dtype)
    BLOCK_M=256
    BLOCK_N=32

    stage = 3 if causal else 1
    grid = lambda META: (
        triton.cdiv(seqlen_q, BLOCK_M),
        batch * nheads,
        1
    )

    _attn_fwd[grid](
        q, k, v, softmax_scale, lse, o,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        batch, nheads,
        N_CTX=seqlen_q,
        BLOCK_DMODEL=d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        pre_load_v=False,
        STAGE=stage,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o, do, delta,
        o.stride(0), o.stride(2), o.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        nheads, seqlen_q, seqlen_q_rounded, d,
        BLOCK_M=64, BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    BLOCK = 64
    grid = (triton.cdiv(seqlen_q, BLOCK), batch * nheads, 1)

    _bwd_kernel_dk_dv[grid](
        q, k, v, softmax_scale,
        do,
        dk, dv,
        lse,
        delta,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        do.stride(0), do.stride(2), do.stride(1), do.stride(3),
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        batch, nheads, seqlen_q,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        BLOCK_DMODEL=d, num_warps=4,
        num_stages=1,
    )
    _bwd_kernel_dq[grid](
        q, k, v, softmax_scale,
        do,
        dq,
        lse,
        delta,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        do.stride(0), do.stride(2), do.stride(1), do.stride(3),
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        batch, nheads, seqlen_q,
        IS_CAUSAL=causal,
        BLOCK_DMODEL=d, num_warps=4,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_stages=1,
    )

class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
        """
            qkv: (batch, seqlen, 3, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
        """
        # Make sure that the last dimension is contiguous
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], bias=bias, causal=causal,
            softmax_scale=softmax_scale
        )
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], 'FlashAttention does not support bias gradient yet'
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(do, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], o, lse,
                                 dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2],
                                 bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return dqkv, None, None, None


flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply


class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
            q: (batch, seqlen_q, nheads, headdim)
            kv: (batch, seqlen_k, 2, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, bias = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], 'FlashAttention does not support bias gradient yet'
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(do, q, kv[:, :, 0], kv[:, :, 1], o, lse,
                                 dq, dkv[:, :, 0], dkv[:, :, 1],
                                 bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return dq, dkv, None, None, None


flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], 'FlashAttention does not support bias gradient yet'
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv,
                                 bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply