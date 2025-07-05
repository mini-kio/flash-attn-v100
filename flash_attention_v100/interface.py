# interface.py
"""
User interface for Flash Attention V100

This module provides the main API that users will interact with.
"""

import torch
from typing import Optional, Tuple, Union
import warnings

from .utils import validate_inputs, calculate_memory_requirements
from .config import get_default_config, DebugConfig
from .ops.attention import FlashAttentionV100Function

def flash_attention_v100(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    block_size_m: Optional[int] = None,
    block_size_n: Optional[int] = None,
    enable_autotuning: bool = True,
    return_softmax_lse: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash Attention implementation optimized for V100 GPUs using Triton
    
    This function implements the Flash Attention algorithm that reduces memory
    usage from O(N²) to O(N) while maintaining mathematical equivalence to
    standard attention.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]  
        v: Value tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
        causal: If True, apply causal masking (lower triangular)
        scale: Attention scale factor. If None, defaults to 1/sqrt(head_dim)
        block_size_m: Block size for M dimension (Q sequence length)
        block_size_n: Block size for N dimension (K/V sequence length)
        enable_autotuning: Whether to use autotuning for optimal performance
        return_softmax_lse: Whether to return log-sum-exp values for debugging
        
    Returns:
        Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        If return_softmax_lse=True, also returns LSE tensor
        
    Example:
        >>> import torch
        >>> from flash_attention_v100 import flash_attention_v100
        >>> 
        >>> batch_size, seq_len, num_heads, head_dim = 2, 1024, 12, 64
        >>> q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> 
        >>> # Standard attention
        >>> output = flash_attention_v100(q, k, v)
        >>> 
        >>> # Causal attention for autoregressive models
        >>> output = flash_attention_v100(q, k, v, causal=True)
    """
    
    # Validate inputs
    q, k, v, scale = validate_inputs(q, k, v, scale)
    
    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    _, seq_len_k, num_heads_kv, _ = k.shape
    
    # Log memory requirements if debugging enabled
    if DebugConfig.LOG_MEMORY_USAGE:
        memory_info = calculate_memory_requirements(
            batch_size, max(seq_len_q, seq_len_k), num_heads_q, head_dim, q.dtype
        )
        print(f"Memory requirements: {memory_info}")
    
    # Get optimal configuration
    if block_size_m is None or block_size_n is None:
        config = get_default_config(max(seq_len_q, seq_len_k), head_dim)
        if block_size_m is None:
            block_size_m = config['BLOCK_M']
        if block_size_n is None:
            block_size_n = config['BLOCK_N']
    
    # Check for potential memory issues
    if seq_len_q > 8192 or seq_len_k > 8192:
        warnings.warn(
            f"Large sequence lengths detected (Q: {seq_len_q}, K: {seq_len_k}). "
            "Consider using gradient checkpointing to reduce memory usage."
        )
    
    # Call the actual implementation
    result = FlashAttentionV100Function.apply(
        q, k, v, scale, causal, block_size_m, block_size_n, 
        enable_autotuning, return_softmax_lse
    )
    
    return result

def flash_attention_v100_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    causal: bool = False,
    scale: Optional[float] = None,
    block_size_m: Optional[int] = None,
    block_size_n: Optional[int] = None,
) -> torch.Tensor:
    """
    Flash Attention for variable-length sequences (packed format)
    
    This function handles sequences of different lengths efficiently by
    packing them together and using cumulative sequence length arrays.
    
    Args:
        q: Packed query tensor [total_q_tokens, num_heads, head_dim]
        k: Packed key tensor [total_kv_tokens, num_heads_kv, head_dim]
        v: Packed value tensor [total_kv_tokens, num_heads_kv, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_kv: Cumulative sequence lengths for keys/values [batch_size + 1]
        max_seqlen_q: Maximum sequence length in query batch
        max_seqlen_kv: Maximum sequence length in key/value batch
        causal: If True, apply causal masking
        scale: Attention scale factor
        block_size_m: Block size for M dimension
        block_size_n: Block size for N dimension
        
    Returns:
        Output tensor [total_q_tokens, num_heads, head_dim]
    """
    
    # TODO: Implement variable-length attention
    # This is a more advanced feature that can be added later
    raise NotImplementedError("Variable-length attention not yet implemented")

def benchmark_flash_attention(
    batch_size: int = 2,
    seq_len: int = 1024, 
    num_heads: int = 12,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    num_warmup: int = 10,
    num_runs: int = 100
) -> dict:
    """
    Benchmark Flash Attention performance
    
    Args:
        batch_size: Batch size for benchmark
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type for tensors
        causal: Whether to use causal attention
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Create random tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        _ = flash_attention_v100(q, k, v, causal=causal)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        output = flash_attention_v100(q, k, v, causal=causal)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    # Calculate memory usage
    memory_info = calculate_memory_requirements(batch_size, seq_len, num_heads, head_dim, dtype)
    
    # Calculate throughput
    total_flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim  # Approximate
    throughput_tflops = (total_flops / (avg_time_ms / 1000)) / 1e12
    
    return {
        'avg_time_ms': avg_time_ms,
        'throughput_tflops': throughput_tflops,
        'memory_usage_mb': memory_info['total_required_mb'],
        'memory_savings_ratio': memory_info['memory_savings_ratio'],
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_heads': num_heads, 
            'head_dim': head_dim,
            'dtype': str(dtype),
            'causal': causal
        }
    }

# Export main functions
__all__ = [
    'flash_attention_v100',
    'flash_attention_v100_varlen', 
    'benchmark_flash_attention'
]