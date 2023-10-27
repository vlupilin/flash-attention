import math
import torch
import torch.nn.functional as F
import subprocess

from datetime import date
from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward
from flash_attn.bert_padding import unpad_input
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

import os
import xlwt


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
    return output.to(dtype=qkv.dtype)

fa_commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
ck_commit = subprocess.run("cd ./csrc/flash_attn_rocm/composable_kernel && git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
datetime = date.today()
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('flash attention')
labels = ["dtype", "batch size", "embedding size", "nheads", "embedding dim", "seqlen", "causal", "dropout", "mi250 fwd(ms)", "mi250 bwd(ms)", "fwd tflops", "bwd tflops", "fwd+bwd tflops"]
for i, label in enumerate(labels):
    worksheet.write(0, i, label = label)


n = 2048
seqlens = [512, 1024, 2048, 4096, 8192, 16384]
# seqlens = [512]
nheads_vals = [16, 32]
# nheads_vals = [16]
i=1
for dtype in [torch.float16, torch.bfloat16]:
    for seqlen in seqlens:
        batch_size = 16384 // seqlen
        for nheads in nheads_vals:
            for causal in [True, False]:
                for dropout_p in [0, 0.17]:
                    torch.manual_seed(0)
                    repeats = 30
                    d = n // nheads
                    print(dtype, batch_size, seqlen, nheads, d, causal, dropout_p)
                    x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
                    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device='cuda', dtype=dtype)

                    lengths = torch.ones(batch_size, 1, device='cuda') * seqlen
                    attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) < lengths

                    x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask_bool)
                    q, k, v = Wqkv(x_unpad).chunk(3, dim=-1)
                    print(q.shape)
                    q = rearrange(q, '(b s) (h d) -> (b s) h d', s=seqlen, h=nheads).detach().contiguous().requires_grad_()
                    k = rearrange(k, '(b s) (h d) -> (b s) h d', s=seqlen, h=nheads).detach().contiguous().requires_grad_()
                    v = rearrange(v, '(b s) (h d) -> (b s) h d', s=seqlen, h=nheads).detach().contiguous().requires_grad_()
                    print(q.shape)
                    # cu_seqlens = cu_seqlens.cpu()
                    fn = lambda q, k, v: flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen_in_batch, max_seqlen_in_batch, dropout_p=dropout_p, causal=causal)
                    t, m1 = benchmark_forward(fn, q, k, v, repeats=repeats, desc='FlashAttention')
                    t, m2 = benchmark_backward(fn, q, k, v, repeats=repeats, desc='FlashAttention')
                    fwd_time = m1.mean * 1000
                    bwd_time = m2.mean * 1000
                    fwd_flops = (0.5 if causal else 1) * 4 * max_seqlen_in_batch * max_seqlen_in_batch * nheads * d * batch_size / 1000000000
                    fwd_tflops = fwd_flops / fwd_time
                    bwd_tflops = 2.5 * fwd_flops / bwd_time
                    fwd_bwd_tflops = 3.5 * fwd_flops / (fwd_time + bwd_time)
                    for j, (label, value) in enumerate(zip(labels, [dtype, batch_size, n, nheads, d, seqlen, causal, dropout_p, format(fwd_time, ".2f"), format(bwd_time, ".2f"), format(fwd_tflops, ".2f"), format(bwd_tflops, ".2f"), format(fwd_bwd_tflops, ".2f")])):
                        worksheet.write(i, j, label = str(value))
                    i += 1

# path = f"./logs/{str(datetime)}/{fa_commit}/{ck_commit}"
# if not os.path.exists(path):
#     os.makedirs(path)

workbook.save(f'performance_grouped.xls')


