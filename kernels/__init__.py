"""
Kernels package for Flash Attention V100

This package contains all Triton kernels for Flash Attention implementation.
"""

from .forward_kernel import (
    flash_attention_forward_triton,
    flash_attention_forward_autotuned,
)

from .backward_kernel import (
    flash_attention_backward_triton,
)

try:
    from .autotuning import (
        get_autotuning_config,
        benchmark_kernel_configs,
        AutoTuner,
    )
    HAS_AUTOTUNING = True
except ImportError:
    HAS_AUTOTUNING = False

from .utils import (
    get_kernel_metadata,
    validate_kernel_config,
    estimate_kernel_memory_usage,
)

__all__ = [
    # Forward kernels
    'flash_attention_forward_triton',
    'flash_attention_forward_autotuned',
    
    # Backward kernels
    'flash_attention_backward_triton',
    
    # Autotuning (if available)
    'get_autotuning_config',
    'benchmark_kernel_configs', 
    'AutoTuner',
    'HAS_AUTOTUNING',
    
    # Utilities
    'get_kernel_metadata',
    'validate_kernel_config',
    'estimate_kernel_memory_usage',
]

# Version information
__version__ = "0.1.0"

# Kernel configuration defaults
DEFAULT_KERNEL_CONFIG = {
    'BLOCK_M': 64,
    'BLOCK_N': 64,
    'BLOCK_K': 64,
    'NUM_STAGES': 3,
    'NUM_WARPS': 4,
}

# V100 specific optimizations
V100_OPTIMIZATIONS = {
    'use_tensor_cores': True,  # V100 has 1st gen Tensor Cores for FP16!
    'tensor_core_precision': 'fp16',  # V100 Tensor Cores support FP16
    'wmma_tile_sizes': [16, 32],  # Supported WMMA tile sizes
    'shared_memory_usage': 'aggressive',
    'register_spilling': 'minimize',
    'memory_coalescing': 'optimize',
}

def get_default_config():
    """Get default kernel configuration for V100"""
    return DEFAULT_KERNEL_CONFIG.copy()

def get_v100_optimizations():
    """Get V100 specific optimization flags"""
    return V100_OPTIMIZATIONS.copy()