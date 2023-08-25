#!/usr/bin/env python3
from functools import partial
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_triton


def attention_ref(qkv, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 16, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    dout = torch.randn_like(q)
    # This is from the attn algorithm
    sm_scale = q.shape[-1] ** (-0.5)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = flash_attn_triton(q, k, v, False, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        assert torch.allclose(ref_dv, tri_dv, atol=0.5, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)


torch.manual_seed(0)
repeats = 10

dropout_p = 0.1
causal = True
dtype = torch.float16
device = 'cuda'

# table 1
print(f'fwd-bs4-nheads48-d64-causal={causal}')
batch_size = [4]
nheads = 48
seqlen = [1024,2048,4096,8192,16384]
n = 3072
d = n // nheads # 64


for bs in batch_size:
    for sq in seqlen:
        if (bs > 32 and sq > 2048) or (bs > 64 and sq > 1024):
           continue
        q = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = q.shape[-1] ** (-0.5)
        qkv_triton = torch.stack([q, k, v], dim=2)

        ## Triton FA implementation
        fn = lambda flash_triton:flash_attn_triton(q, k, v, causal, 1.3)
        fa_time,fa_measurement = benchmark_combined(fn, qkv_triton, repeats=repeats, desc='FlashAttention triton', verbose=False)

        ## Measure time and compute tflops
        triton_time = fa_measurement.mean;
        # 5x because the backward pass has 5 dot products, all of the same size
        # 2x because each MAC has 2 ops.
        flops_per_matmul = 7. * bs * nheads * sq * sq * d
        total_flops = 2 * flops_per_matmul
        if causal:
           total_flops *= .5
        triton_tflops = total_flops / triton_time * 1e-12
        print(f'{bs:3d}  {sq:10d} {triton_tflops:.2f} tflops {triton_time*1e3:.3f} ms')

# table 2
print(f'fwd-nheads16-d64-causal={causal}')
batch_size = [1,32,48,64,128]
nheads = 16
seqlen = [1024,2048,4096]
n = 1024
d = n // nheads # 64


for bs in batch_size:
    for sq in seqlen:
        if (bs > 32 and sq > 2048) or (bs > 64 and sq > 1024):
            continue
        q = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((bs, nheads, sq, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = q.shape[-1] ** (-0.5)
        qkv_triton = torch.stack([q, k, v], dim=2)

        ## Triton FA implementation
        fn = lambda flash_triton:flash_attn_triton(q, k, v, causal, sm_scale)
        fa_time,fa_measurement = benchmark_combined(fn, qkv_triton, repeats=repeats, desc='FlashAttention triton', verbose=False)

        ## Measure time and compute tflops
        triton_time = fa_measurement.mean;
        # 5x because the backward pass has 5 dot products, all of the same size
        # 2x because each MAC has 2 ops.
        flops_per_matmul = 7. * bs * nheads * sq * sq * d
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= .5
        triton_tflops = total_flops / triton_time * 1e-12
        print(f'{bs:3d}  {sq:10d} {triton_tflops:.2f} tflops {triton_time*1e3:.3f} ms')

