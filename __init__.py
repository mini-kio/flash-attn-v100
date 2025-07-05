# Copyright 2024 Flash Attention V100 Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Flash Attention for V100 GPUs using Triton

This package provides an implementation of Flash Attention specifically 
optimized for V100 GPUs (Compute Capability 7.0) using Triton kernels.

Main Features:
- Memory-efficient attention computation
- Support for causal and non-causal attention
- Gradient computation for training
- Automatic kernel tuning for optimal performance
"""

import torch
try:
    from .config import SUPPORTED_DTYPES, MIN_CUDA_VERSION, V100_COMPUTE_CAPABILITY
    from .interface import flash_attention_v100
    from .ops.attention import FlashAttentionV100Function
except ImportError:
    # Fallback for testing without full package setup
    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
    MIN_CUDA_VERSION = (11, 0)
    V100_COMPUTE_CAPABILITY = (7, 0)
    
    def flash_attention_v100(q, k, v, **kwargs):
        """Simple fallback implementation"""
        scale = kwargs.get('scale', (q.shape[-1] ** -0.5))
        
        # q, k, v shape: [batch, seq_len, num_heads, head_dim]
        # For attention: we need [batch, num_heads, seq_len, head_dim] 
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim] 
        v = v.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Compute attention weights: [batch, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q * scale, k.transpose(-2, -1))
        
        if kwargs.get('causal', False):
            batch_size, num_heads, seq_len, head_dim = q.shape
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            # Expand mask to [batch, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            attn_weights.masked_fill_(mask, float('-inf'))
            
        attn_probs = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Transpose back to original format: [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)
        return output
    
    FlashAttentionV100Function = None

__version__ = "0.1.0"
__author__ = "Flash Attention V100 Team"

# Check CUDA availability and version
def _check_environment():
    """Check if the environment supports Flash Attention V100"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Flash Attention V100 requires CUDA.")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("Could not determine CUDA version.")
    
    major, minor = map(int, cuda_version.split('.')[:2])
    min_major, min_minor = MIN_CUDA_VERSION
    
    if (major, minor) < (min_major, min_minor):
        raise RuntimeError(
            f"CUDA version {cuda_version} is not supported. "
            f"Minimum required version is {min_major}.{min_minor}"
        )
    
    # Check GPU compute capability
    device_capability = torch.cuda.get_device_capability()
    if device_capability < V100_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"GPU compute capability {device_capability} is not supported. "
            f"Minimum required capability is {V100_COMPUTE_CAPABILITY} (V100 or newer)."
        )
    
    return True

# Check environment on import
try:
    _check_environment()
    _ENVIRONMENT_OK = True
except RuntimeError as e:
    print(f"Warning: {e}")
    _ENVIRONMENT_OK = False

# Export main functions
__all__ = [
    "flash_attention_v100",
    "FlashAttentionV100Function", 
    "SUPPORTED_DTYPES",
    "_ENVIRONMENT_OK",
    "__version__"
]

# Convenience imports for users
try:
    from .ops.attention import (
        flash_attention_forward,
        flash_attention_backward,
    )
    
    # Add to __all__
    __all__.extend([
        "flash_attention_forward",
        "flash_attention_backward",
    ])
except ImportError:
    # Skip if ops.attention is not available
    pass