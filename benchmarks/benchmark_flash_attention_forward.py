import pickle
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.utils.benchmark import benchmark_forward
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, "b t h d -> (b h) t d")
    k = rearrange(k, "b s h d -> (b h) d s")
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(
        batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
    )
    scores = rearrange(
        torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
        "(b h) t s -> b h t s",
        h=nheads,
    )
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output.to(dtype=qkv.dtype)


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


repeats = 30
device = "cuda"
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = (
    ["Flash2", "Pytorch"]
    + (["Triton"] if attention_triton is not None else [])
    + (["xformers"] if xops is not None else [])
)

time_f = {}
speed_f = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            nheads_kv = nheads // 4
            qkv = torch.randn(
                 batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype
            )
            q = torch.randn(
                 batch_size, seqlen, nheads, headdim, device=device, dtype=dtype
            )
            k = torch.randn(
                 batch_size, seqlen, nheads, headdim, device=device, dtype=dtype
            )
            v = torch.randn(
                 batch_size, seqlen, nheads, headdim, device=device, dtype=dtype
            )

            f = time_fwd(
                flash_attn_func,
                q,
                k,
                v,
                dropout_p,
                causal=causal,
                repeats=repeats,
                verbose=False,
            )
            time_f[config, "Flash2"] = f

            try:
                qkv = qkv.detach().requires_grad_(False)
                f = time_fwd(
                    attention_pytorch,
                    qkv,
                    dropout_p,
                    causal=causal,
                    repeats=repeats,
                    verbose=False,
                )
            except:  # Skip if OOM
                f = float("nan")
            time_f[config, "Pytorch"] = f

            if attention_triton is not None:
                q, k, v = [
                    torch.randn(
                        batch_size,
                        nheads,
                        seqlen,
                        headdim,
                        device=device,
                        dtype=dtype,
                        requires_grad=False,
                    )
                    for _ in range(3)
                ]
                # Try both values of sequence_parallel and pick the faster one
                try:
                    f = time_fwd(
                        attention_triton,
                        q,
                        k,
                        v,
                        causal,
                        headdim ** (-0.5),
                        False,
                        repeats=repeats,
                        verbose=False,
                    )
                except:
                    f = float("nan")
                try:
                    _ = time_fwd(
                        attention_triton,
                        q,
                        k,
                        v,
                        causal,
                        headdim ** (-0.5),
                        True,
                        repeats=repeats,
                        verbose=False,
                    )
                except:
                    time_f[config, "Triton"] = f

            if xops is not None:
                q, k, v = [
                    torch.randn(
                        batch_size,
                        seqlen,
                        nheads,
                        headdim,
                        device=device,
                        dtype=dtype,
                        requires_grad=False,
                    )
                    for _ in range(3)
                ]
                f = time_fwd(
                    xops.memory_efficient_attention,
                    q,
                    k,
                    v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.cutlass.FwOp),
                )
                time_f[config, "xformers"] = f

            print(
                f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###"
            )
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal),
                    time_f[config, method],
                )
                print(f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s")
