import torch
import torch.nn.functional as F
import functools

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


@functools.lru_cache(maxsize=16)
def gen_cu_seqlen(batch_size, n, device="cuda"):
    cu_seqlens = torch.arange(
            0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=device
        )
    return cu_seqlens


def _flash_attn(q, k, v, mask=None, bias=None, q_cu_seqlens=None, k_cu_seqlens=None):
    batch_dims = q.shape[:-3]
    n, no_heads, c = q.shape[-3:]
    dtype = q.dtype

    # for cross attention
    k_batch_dims = k.shape[:-3]
    k_n, k_no_heads, k_c = k.shape[-3:]

    # [B_flat, N, H, C]
    q = q.view(-1, *q.shape[-3:])
    k = k.view(-1, *k.shape[-3:])
    v = v.view(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    k_batch_size = k.shape[0]
    
    # [B_flat * N, H, C]
    q = q.view(-1, *q.shape[-2:])
    k = k.view(-1, *k.shape[-2:])
    v = v.view(-1, *v.shape[-2:])

    q_max_s = n
    q_cu_seqlens = gen_cu_seqlen(batch_size, q_max_s, q.device)

    k_max_s = k_n
    k_cu_seqlens = gen_cu_seqlen(k_batch_size, k_max_s, k.device)

    if mask is not None:
        mask_heads, tgt_len, src_len = mask.shape[-3:]
        mask = mask.view(-1 , mask_heads, tgt_len, src_len).contiguous()

    if bias is not None:
        bias_heads, tgt_len, src_len = bias.shape[-3:]
        bias = bias.view(-1 , bias_heads, tgt_len, src_len).contiguous()
    
    out = flash_attn_unpadded_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        attn_mask=mask,
        attn_bias=bias,
        dropout_p = 0.,
        softmax_scale = 1., # q has been scaled already
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c)
    return out
