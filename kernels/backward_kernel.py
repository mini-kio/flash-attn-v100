# backward_kernel.py
"""
Flash Attention Backward Kernel for V100 using Triton

This implements the backward pass of Flash Attention with memory efficiency.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _flash_attention_backward_kernel(
    Q, K, V, O, DO, DQ, DK, DV,  # Input, output and gradient pointers
    L, M, Delta,                  # Forward pass statistics and precomputed delta
    seq_len_q, seq_len_k, num_heads, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,   # Q strides
    stride_kb, stride_kh, stride_ks, stride_kd,   # K strides  
    stride_vb, stride_vh, stride_vs, stride_vd,   # V strides
    stride_ob, stride_oh, stride_os, stride_od,   # O strides
    stride_dob, stride_doh, stride_dos, stride_dod, # DO strides
    stride_dqb, stride_dqh, stride_dqs, stride_dqd, # DQ strides
    stride_dkb, stride_dkh, stride_dks, stride_dkd, # DK strides
    stride_dvb, stride_dvh, stride_dvs, stride_dvd, # DV strides
    stride_lb, stride_lh, stride_ls,              # L strides
    stride_mb, stride_mh, stride_ms,              # M strides
    stride_deltab, stride_deltah, stride_deltas,  # Delta strides
    scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Flash Attention Backward Kernel
    
    Computes gradients for Q, K, V using the Flash Attention algorithm
    while maintaining memory efficiency.
    """
    
    # Program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    kv_block_id = tl.program_id(2)
    
    # KV block range
    kv_start = kv_block_id * BLOCK_N
    kv_end = tl.minimum(kv_start + BLOCK_N, seq_len_k)
    
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Base pointers for current batch and head
    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K + batch_id * stride_kb + head_id * stride_kh
    v_base = V + batch_id * stride_vb + head_id * stride_vh
    o_base = O + batch_id * stride_ob + head_id * stride_oh
    do_base = DO + batch_id * stride_dob + head_id * stride_doh
    
    # Output gradient pointers
    dq_base = DQ + batch_id * stride_dqb + head_id * stride_dqh
    dk_base = DK + batch_id * stride_dkb + head_id * stride_dkh
    dv_base = DV + batch_id * stride_dvb + head_id * stride_dvh
    
    # Statistics pointers
    l_base = L + batch_id * stride_lb + head_id * stride_lh
    m_base = M + batch_id * stride_mb + head_id * stride_mh
    delta_base = Delta + batch_id * stride_deltab + head_id * stride_deltah
    
    # Load current K, V blocks
    k_ptr = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptr = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k_block = tl.load(k_ptr, mask=offs_n[:, None] < seq_len_k, other=0.0)
    v_block = tl.load(v_ptr, mask=offs_n[:, None] < seq_len_k, other=0.0)
    
    # Initialize gradient accumulators for current K, V blocks
    dk_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # Iterate over Q blocks
    for q_block_id in range(0, tl.cdiv(seq_len_q, BLOCK_M)):
        q_start = q_block_id * BLOCK_M
        q_end = tl.minimum(q_start + BLOCK_M, seq_len_q)
        
        offs_m = q_start + tl.arange(0, BLOCK_M)
        
        # Load Q block and related data
        q_ptr = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptr = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptr = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        
        q_block = tl.load(q_ptr, mask=offs_m[:, None] < seq_len_q, other=0.0)
        o_block = tl.load(o_ptr, mask=offs_m[:, None] < seq_len_q, other=0.0)
        do_block = tl.load(do_ptr, mask=offs_m[:, None] < seq_len_q, other=0.0)
        
        # Load statistics
        l_ptr = l_base + offs_m * stride_ls
        m_ptr = m_base + offs_m * stride_ms
        delta_ptr = delta_base + offs_m * stride_deltas
        
        l_i = tl.load(l_ptr, mask=offs_m < seq_len_q, other=0.0)
        m_i = tl.load(m_ptr, mask=offs_m < seq_len_q, other=0.0)
        delta_i = tl.load(delta_ptr, mask=offs_m < seq_len_q, other=0.0)
        
        # Compute attention scores: Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q_block, tl.trans(k_block))
        qk *= scale
        
        # Apply causal mask if needed
        if causal:
            causal_mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, -float('inf'))
        
        # Compute attention probabilities
        p = tl.exp(qk - m_i[:, None])
        p = p / l_i[:, None]
        
        # Compute dP (gradient w.r.t. attention probabilities)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do_block, tl.trans(v_block))
        dp = (dp - delta_i[:, None]) * p
        
        # Apply causal mask to gradients
        if causal:
            dp = tl.where(causal_mask, dp, 0.0)
        
        # Compute gradients for current blocks
        # dV = P^T @ dO  
        dv_acc += tl.dot(tl.trans(p.to(v_block.dtype)), do_block)
        
        # dK = scale * dP^T @ Q
        dk_acc += scale * tl.dot(tl.trans(dp.to(k_block.dtype)), q_block)
        
        # dQ = scale * dP @ K
        dq = scale * tl.dot(dp.to(k_block.dtype), k_block)
        
        # Store dQ for current block (accumulate if multiple KV blocks)
        dq_ptr = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        if kv_block_id == 0:
            # First KV block - initialize DQ
            tl.store(dq_ptr, dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < seq_len_q)
        else:
            # Subsequent KV blocks - accumulate DQ
            existing_dq = tl.load(dq_ptr, mask=offs_m[:, None] < seq_len_q, other=0.0)
            new_dq = existing_dq + dq.to(DQ.dtype.element_ty)
            tl.store(dq_ptr, new_dq, mask=offs_m[:, None] < seq_len_q)
    
    # Store accumulated gradients for K, V
    dk_ptr = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptr = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptr, dk_acc.to(DK.dtype.element_ty), mask=offs_n[:, None] < seq_len_k)
    tl.store(dv_ptr, dv_acc.to(DV.dtype.element_ty), mask=offs_n[:, None] < seq_len_k)


@triton.jit
def _precompute_delta_kernel(
    O, DO, Delta,
    seq_len, num_heads, head_dim,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_deltab, stride_deltah, stride_deltas,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Precompute delta = sum(O * dO, dim=-1) for backward pass
    
    This is needed for the Flash Attention backward algorithm.
    """
    
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_block_id = tl.program_id(2)
    
    seq_start = seq_block_id * BLOCK_M
    offs_m = seq_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers to O and dO
    o_ptr = (O + batch_id * stride_ob + head_id * stride_oh + 
             offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    do_ptr = (DO + batch_id * stride_dob + head_id * stride_doh +
              offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod)
    
    # Load O and dO blocks
    o_block = tl.load(o_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    do_block = tl.load(do_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    
    # Compute delta = sum(O * dO, dim=-1)
    delta = tl.sum(o_block * do_block, 1)
    
    # Store delta
    delta_ptr = (Delta + batch_id * stride_deltab + head_id * stride_deltah +
                 offs_m * stride_deltas)
    tl.store(delta_ptr, delta, mask=offs_m < seq_len)


def flash_attention_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    max_vals: torch.Tensor,
    scale: float,
    causal: bool = False,
    block_size_m: int = 64,
    block_size_n: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash Attention backward pass using Triton kernel
    
    Args:
        q: Query tensor from forward pass
        k: Key tensor from forward pass
        v: Value tensor from forward pass
        o: Output tensor from forward pass
        do: Gradient w.r.t. output
        lse: Log-sum-exp values from forward pass
        max_vals: Max values from forward pass
        scale: Attention scale factor
        causal: Whether causal masking was used
        block_size_m: Block size for sequence dimension (queries)
        block_size_n: Block size for sequence dimension (keys/values)
        
    Returns:
        Tuple of (dq, dk, dv) gradient tensors
    """
    
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    
    # Allocate gradient tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    
    # Precompute delta = sum(O * dO, dim=-1)
    delta = torch.empty((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)
    
    # Launch delta precomputation kernel
    grid_delta = (batch_size, num_heads, triton.cdiv(seq_len_q, block_size_m))
    _precompute_delta_kernel[grid_delta](
        o, do, delta,
        seq_len_q, num_heads, head_dim,
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        do.stride(0), do.stride(2), do.stride(1), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        BLOCK_M=block_size_m,
        BLOCK_DMODEL=head_dim,
        num_warps=4,
    )
    
    # Launch backward kernel
    grid = (batch_size, num_heads, triton.cdiv(seq_len_k, block_size_n))
    _flash_attention_backward_kernel[grid](
        q, k, v, o, do, dq, dk, dv,
        lse, max_vals, delta,
        seq_len_q, seq_len_k, num_heads, head_dim,
        # Q strides
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        # K strides
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        # V strides  
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        # O strides
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        # dO strides
        do.stride(0), do.stride(2), do.stride(1), do.stride(3),
        # dQ strides
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        # dK strides
        dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3),
        # dV strides
        dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3),
        # LSE strides
        lse.stride(0), lse.stride(1), lse.stride(2),
        # Max strides
        max_vals.stride(0), max_vals.stride(1), max_vals.stride(2),
        # Delta strides
        delta.stride(0), delta.stride(1), delta.stride(2),
        scale, causal,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_DMODEL=head_dim,
        num_warps=4,
        num_stages=3,
    )
    
    return dq, dk, dv


__all__ = [
    'flash_attention_backward_triton',
]