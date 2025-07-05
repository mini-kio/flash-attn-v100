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
from .config import SUPPORTED_DTYPES, MIN_CUDA_VERSION, V100_COMPUTE_CAPABILITY
from .interface import flash_attention_v100
from .ops.attention import FlashAttentionV100Function

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
from .ops.attention import (
    flash_attention_forward,
    flash_attention_backward,
)

# Add to __all__
__all__.extend([
    "flash_attention_forward",
    "flash_attention_backward",
])