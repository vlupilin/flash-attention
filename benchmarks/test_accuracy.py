import math
import torch
import torch.nn.functional as F

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_triton


def attention_ref(q,k,v, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    # q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = q.shape[1]
    d = q.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=q.dtype)


torch.manual_seed(0)
repeats = 30
batch_size = 64
nheads = 16
seqlen = 1024
n = 1024
d = n // nheads
dropout_p = 0
causal = False
dtype = torch.float16
device = 'cuda'

q = torch.empty((batch_size, nheads, seqlen, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
k = torch.empty((batch_size, nheads, seqlen, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
v = torch.empty((batch_size, nheads, seqlen, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()

fn = lambda q, k, v: flash_attn_triton(
    q, k, v, causal, sm_scale = q.shape[-1] ** (-0.5)
)
out1 = fn(q, k, v)
# benchmark_all(fn, q, k, v, repeats=repeats, desc='FlashAttention')
fn = lambda q,k,v: attention_ref(q,k,v, torch.ones(batch_size, seqlen, dtype=torch.bool).cuda(), dropout_p, causal=causal)
out2 = fn(q.transpose(1, 2),k.transpose(1, 2),v.transpose(1, 2))
torch.testing.assert_close(out1, out2.transpose(1, 2), atol=1e-4, rtol=1e-4)