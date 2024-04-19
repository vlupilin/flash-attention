# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import xlwt

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(4, 1024), (4, 2048), (4, 4096), (4, 8192), (4, 16384)]
causal_vals = [False, True]
headdim_vals = [64]
dim = 3072
dropout_p = 0.0

methods = (["Flash2"])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

work_book = xlwt.Workbook(encoding='utf-8')
sheet_data = work_book.add_sheet('fla')
sheet_data.write(0, 0, "batch_size")
sheet_data.write(0, 1, "nhead")
sheet_data.write(0, 2, "seqlen")
sheet_data.write(0, 3, "head_dim")
sheet_data.write(0, 4, "mode")
sheet_data.write(0, 5, "causal")
sheet_data.write(0, 6, "TFLOPS")

i=1
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            f, b = time_fwd_bwd(
                flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "Flash2"] = f
            time_b[config, "Flash2"] = b

            work_book = xlwt.Workbook(encoding='utf-8')
            sheet_data = work_book.add_sheet('fla')

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                )

                sheet_data.write(i, 0, batch_size)
                sheet_data.write(i, 1, nheads)
                sheet_data.write(i, 2, seqlen)
                sheet_data.write(i, 3, headdim)
                sheet_data.write(i, 4, "fwd")
                sheet_data.write(i, 5, causal)
                sheet_data.write(i, 6, speed_f[config, method])

                sheet_data.write(i+1, 0, batch_size)
                sheet_data.write(i+1, 1, nheads)
                sheet_data.write(i+1, 2, seqlen)
                sheet_data.write(i+1, 3, headdim)
                sheet_data.write(i+1, 4, "bwd")
                sheet_data.write(i+1, 5, causal)
                sheet_data.write(i+1, 6, speed_b[config, method])

                i = i+2


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
