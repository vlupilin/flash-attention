
import torch
import subprocess

from datetime import date
from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward
from flash_attn.bert_padding import unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_func

import xlwt
from absl import flags

flags.DEFINE_string('logdir', './', 'log directory.')
FLAGS = flags.FLAGS

fa_commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
ck_commit = subprocess.run("cd ./csrc/flash_attn_rocm/composable_kernel && git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
datetime = date.today()
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('flash attention')
labels = ["dtype", "batch size", "embedding size", "nheads", "embedding dim", "seqlen", "casual", "dropout", "mi250 fwd(ms)", "mi250 bwd(ms)", "mi250 torch fwd(ms)", "mi250 torch bwd(ms)"]
for i, label in enumerate(labels):
    worksheet.write(0, i, label = label)

item_lists=[
[16  ,  768   ,   12   ,  2048] ,
[32  ,  768   ,   12   ,  2048] ,
[8   ,  1024  ,   16   ,  2048] ,
[16  ,  1024  ,   16   ,  2048] ,
[4   ,  1536  ,   12   ,  2048] ,
[8   ,  1536  ,   12   ,  2048] ,
[16  ,  2048  ,   16   ,  2048] ,
[32  ,  2048  ,   16   ,  2048] ,
[8   ,  2560  ,   20   ,  2048] ,
[16  ,  2560  ,   20   ,  2048] ,
[8   ,  4096  ,   32   ,  2048] ,
[16  ,  4096  ,   32   ,  2048] ,
[8   ,  5120  ,   40   ,  2048] ,
[16  ,  5120  ,   40   ,  2048] ,
[8   ,  7168  ,   56   ,  2048] ,
[16  ,  7168  ,   56   ,  2048] ,
[4   ,  8192  ,   64   ,  2048] ,
[8   ,  8192  ,   64   ,  2048]
]


i = 1
for dtype in [torch.float16, torch.bfloat16]:
    for batch_size, n, nheads, seqlen in item_lists:
        for causal in [True, False]:
            for dropout_p in [0, 0.17]:
                torch.manual_seed(0)
                repeats = 30
                d = n // nheads
                device = 'cuda'
                print(dtype, batch_size, n, nheads, d, causal, dropout_p)
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
                t,m2 = benchmark_backward(fn, q, k, v, repeats=repeats, desc='FlashAttention')

                for j, (label, value) in enumerate(zip(labels, [dtype, batch_size, n, nheads, d, seqlen, causal, dropout_p, format(m1.mean*1000, ".2f"), format(m2.mean*1000, ".2f"), 'None', 'None'])):
                    worksheet.write(i, j, label = str(value))
                i += 1

workbook.save(f'{FLAGS.logdir}/{str(datetime)}/{fa_commit}/{ck_commit}/performance.xls')