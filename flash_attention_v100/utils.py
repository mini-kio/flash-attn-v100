# utils.py
"""
Utility functions for Flash Attention V100
"""

import torch
import math
from typing import Tuple, Optional, Union
from .config import SUPPORTED_DTYPES, NumericalConfig

def validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                   scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Validate and prepare input tensors for Flash Attention
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len, num_heads, head_dim] 
        v: Value tensor [batch_size, seq_len, num_heads, head_dim]
        scale: Attention scale factor
        
    Returns:
        Validated and prepared q, k, v tensors and scale factor
    """
    # Check tensor shapes
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Input tensors must be 4-dimensional [batch, seq_len, num_heads, head_dim]")
    
    batch_size, seq_len_q, num_heads_q, head_dim_q = q.shape
    batch_size_k, seq_len_k, num_heads_k, head_dim_k = k.shape
    batch_size_v, seq_len_v, num_heads_v, head_dim_v = v.shape
    
    # Check batch sizes match
    if not (batch_size == batch_size_k == batch_size_v):
        raise ValueError("Batch sizes must match across Q, K, V")
    
    # Check sequence lengths match for K and V
    if seq_len_k != seq_len_v:
        raise ValueError("Sequence lengths of K and V must match")
    
    # Check head dimensions match
    if not (head_dim_q == head_dim_k == head_dim_v):
        raise ValueError("Head dimensions must match across Q, K, V")
    
    # Check number of heads (support for grouped attention)
    if num_heads_q % num_heads_k != 0:
        raise ValueError("Number of query heads must be divisible by number of key/value heads")
    
    # Check data types
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("All tensors must have the same dtype")
    
    if q.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {q.dtype}. Supported types: {SUPPORTED_DTYPES}")
    
    # Check device
    if not (q.device == k.device == v.device):
        raise ValueError("All tensors must be on the same device")
    
    if not q.device.type == 'cuda':
        raise ValueError("Tensors must be on CUDA device")
    
    # Set default scale
    if scale is None:
        scale = head_dim_q ** NumericalConfig.DEFAULT_SCALE_POWER
    
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous() 
    v = v.contiguous()
    
    return q, k, v, scale

def get_tensor_info(tensor: torch.Tensor) -> dict:
    """Get comprehensive information about a tensor"""
    return {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'is_contiguous': tensor.is_contiguous(),
        'memory_usage_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
        'requires_grad': tensor.requires_grad
    }

def calculate_memory_requirements(batch_size: int, seq_len: int, num_heads: int, 
                                head_dim: int, dtype: torch.dtype) -> dict:
    """
    Calculate memory requirements for Flash Attention
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        
    Returns:
        Dictionary with memory requirement information
    """
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Input tensors memory (Q, K, V)
    qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * element_size
    
    # Output tensor memory
    output_memory = batch_size * seq_len * num_heads * head_dim * element_size
    
    # Intermediate attention matrix (without Flash Attention optimization)
    # This is what we're trying to avoid!
    full_attention_memory = batch_size * num_heads * seq_len * seq_len * element_size
    
    # Flash Attention intermediate memory (much smaller blocks)
    from .config import BlockConfig
    block_m = BlockConfig.DEFAULT_BLOCK_M
    block_n = BlockConfig.DEFAULT_BLOCK_N
    flash_intermediate_memory = batch_size * num_heads * block_m * block_n * element_size
    
    return {
        'qkv_memory_mb': qkv_memory / (1024 * 1024),
        'output_memory_mb': output_memory / (1024 * 1024),
        'full_attention_memory_mb': full_attention_memory / (1024 * 1024),
        'flash_intermediate_memory_mb': flash_intermediate_memory / (1024 * 1024),
        'memory_savings_ratio': full_attention_memory / flash_intermediate_memory,
        'total_required_mb': (qkv_memory + output_memory + flash_intermediate_memory) / (1024 * 1024)
    }

def get_optimal_block_size(seq_len: int, head_dim: int, available_memory: int) -> Tuple[int, int]:
    """
    Determine optimal block sizes based on sequence length and available memory
    
    Args:
        seq_len: Sequence length
        head_dim: Head dimension  
        available_memory: Available shared memory in bytes
        
    Returns:
        Tuple of (block_m, block_n) sizes
    """
    from .config import BlockConfig, V100MemoryHierarchy
    
    # Start with default block sizes
    block_m = BlockConfig.DEFAULT_BLOCK_M
    block_n = BlockConfig.DEFAULT_BLOCK_N
    
    # Calculate memory usage per block
    element_size = 2  # Assume fp16 for calculation
    memory_per_block = block_m * block_n * head_dim * element_size * 3  # Q, K, V blocks
    
    # Adjust if exceeding available memory
    max_memory = min(available_memory, V100MemoryHierarchy.SRAM_SIZE // 2)  # Use half of SRAM
    
    while memory_per_block > max_memory and block_m > 32:
        block_m = max(32, block_m // 2)
        block_n = max(32, block_n // 2) 
        memory_per_block = block_m * block_n * head_dim * element_size * 3
    
    return block_m, block_n

def cdiv(a: int, b: int) -> int:
    """Ceiling division"""
    return (a + b - 1) // b

def next_power_of_2(x: int) -> int:
    """Get the next power of 2 greater than or equal to x"""
    return 2 ** math.ceil(math.log2(x))

def is_power_of_2(x: int) -> bool:
    """Check if x is a power of 2"""
    return x > 0 and (x & (x - 1)) == 0

def pad_to_multiple(x: int, multiple: int) -> int:
    """Pad x to the next multiple"""
    return cdiv(x, multiple) * multiple

def log_memory_usage(tensor_dict: dict, prefix: str = ""):
    """Log memory usage of tensors"""
    total_memory = 0
    print(f"{prefix}Memory Usage:")
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            total_memory += memory_mb
            print(f"  {name}: {memory_mb:.2f} MB, shape: {tensor.shape}")
    print(f"  Total: {total_memory:.2f} MB")

def check_tensor_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Check for NaN or Inf values in tensor"""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

# Export utility functions
__all__ = [
    'validate_inputs',
    'get_tensor_info', 
    'calculate_memory_requirements',
    'get_optimal_block_size',
    'cdiv',
    'next_power_of_2',
    'is_power_of_2', 
    'pad_to_multiple',
    'log_memory_usage',
    'check_tensor_nan_inf'
]