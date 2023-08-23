from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

from datetime import date
from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_func

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
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)

fa_commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
ck_commit = subprocess.run("cd ./csrc/flash_attn_rocm/composable_kernel && git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
datetime = date.today()
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('flash attention')
labels = ["dtype", "batch size", "embedding size", "nheads", "embedding dim", "seqlen", "casual", "dropout", "mi250 fwd(ms)", "mi250 bwd(ms)", "mi250 torch fwd(ms)", "mi250 torch bwd(ms)"]
for i, label in enumerate(labels):
    worksheet.write(0, i, label = label)

item_lists=[
[32  ,  2048  ,   32   ,  512  ],
[16  ,  2048  ,   32   ,  1024 ],
[8   ,  2048  ,   32   ,  2048 ],
[4   ,  2048  ,   32   ,  4096 ],
[2   ,  2048  ,   32   ,  8192 ],
[1   ,  2048  ,   32   ,  16384],
[32  ,  2048  ,   16   ,  512  ],
[16  ,  2048  ,   16   ,  1024 ],
[8   ,  2048  ,   16   ,  2048 ],
[4   ,  2048  ,   16   ,  4096 ],
[2   ,  2048  ,   16   ,  8192 ],
[1   ,  2048  ,   16   ,  16384]
]


i=1
for dtype in [torch.float16, torch.bfloat16]:
    for batch_size, n, nheads, seqlen in item_lists:
        for causal in [True, False]:
            for dropout_p in [0]:
                torch.manual_seed(0)
                repeats = 30
                # n = nheads * d
                d = n // nheads
                device = 'cuda'
                print(batch_size, n, nheads, d)
                x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
                Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

                lengths = torch.ones(batch_size, 1, device='cuda') * seqlen
                attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) < lengths
                attention_mask = torch.zeros(batch_size, seqlen, device='cuda', dtype=dtype)
                attention_mask[~attention_mask_bool] = -10000.0
                attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

                x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask_bool)
                q, k, v = Wqkv(x_unpad).chunk(3, dim=-1)
                q = rearrange(q, 'nnz (h d) -> nnz h d', h=nheads).detach().contiguous().requires_grad_()
                k = rearrange(k, 'nnz (h d) -> nnz h d', h=nheads).detach().contiguous().requires_grad_()
                v = rearrange(v, 'nnz (h d) -> nnz h d', h=nheads).detach().contiguous().requires_grad_()
                cu_seqlens = cu_seqlens.cpu()
                fn = lambda q, k, v: flash_attn_unpadded_func(
                    q, k, v, cu_seqlens, cu_seqlens, max_seqlen_in_batch, max_seqlen_in_batch, dropout_p, causal=causal
                )
                t,m1 = benchmark_forward(fn, q, k, v, repeats=repeats, desc='FlashAttention')
                #t,m2 = benchmark_backward(fn, q, k, v, repeats=repeats, desc='FlashAttention')
                # fn = lambda q, k, v: attention_ref(q, k, v, attention_mask_bool, dropout_p, causal=causal)
                # try:
                #     t,m3 = benchmark_forward(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')
                #     t,m4 = benchmark_backward(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')
                # except:
                #     for j, (label, value) in enumerate(zip(labels, [dtype, batch_size, n, nheads, d, seqlen, causal, dropout_p, "OOM", "OOM", "OOM", "OOM"])):
                #         worksheet.write(i, j, label = str(value))
                #     i += 1
                #     continue
                for j, (label, value) in enumerate(zip(labels, [dtype, batch_size, n, nheads, d, seqlen, causal, dropout_p, format(m1.mean*1000, ".2f"), 'None', 'None', 'None'])):
                    worksheet.write(i, j, label = str(value))
                i += 1
                #exit(0)
workbook.save(f'./performance.xls')
# fn = lambda qkv: attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)
# benchmark_all(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')
