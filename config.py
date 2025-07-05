# config.py
"""
Configuration and constants for Flash Attention V100
"""

import torch

# GPU Requirements
V100_COMPUTE_CAPABILITY = (7, 0)  # V100 minimum compute capability
MIN_CUDA_VERSION = (11, 0)  # Minimum CUDA version

# Supported data types
SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}

# V100 specific memory hierarchy (from the provided diagram)
class V100MemoryHierarchy:
    """V100 GPU memory hierarchy specifications"""
    
    # SRAM (Shared Memory) specifications
    SRAM_BANDWIDTH = 19 * 1024**4  # 19 TB/s in bytes/s
    SRAM_SIZE = 96 * 1024  # 96KB shared memory per SM
    
    # HBM (High Bandwidth Memory) specifications  
    HBM_BANDWIDTH = 900 * 1024**3  # ~900 GB/s for V100
    HBM_SIZE = 16 * 1024**3  # 16GB for V100
    
    # DRAM (if applicable)
    DRAM_BANDWIDTH = 12.8 * 1024**3  # 12.8 GB/s

# Block size configurations for different scenarios
class BlockConfig:
    """Block size configurations optimized for V100 with Tensor Cores"""
    
    # Default block sizes (optimized for V100 Tensor Cores - 16의 배수)
    DEFAULT_BLOCK_M = 64
    DEFAULT_BLOCK_N = 64
    DEFAULT_BLOCK_K = 64
    
    # Small sequence length configurations (Tensor Core optimized)
    SMALL_SEQ_CONFIG = {
        'BLOCK_M': 32,
        'BLOCK_N': 32, 
        'BLOCK_K': 32,
        'NUM_STAGES': 2,
        'NUM_WARPS': 4,
    }
    
    # Medium sequence length configurations (Tensor Core optimized)
    MEDIUM_SEQ_CONFIG = {
        'BLOCK_M': 64,
        'BLOCK_N': 64,
        'BLOCK_K': 64, 
        'NUM_STAGES': 3,
        'NUM_WARPS': 4,
    }
    
    # Large sequence length configurations (Tensor Core optimized)
    LARGE_SEQ_CONFIG = {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
        'BLOCK_K': 64,
        'NUM_STAGES': 4,
        'NUM_WARPS': 8,
    }
    
    # V100 Tensor Core specific configurations
    TENSOR_CORE_CONFIG = {
        'BLOCK_M': 64,  # 16의 배수
        'BLOCK_N': 64,  # 16의 배수
        'BLOCK_K': 64,  # 16의 배수
        'NUM_STAGES': 3,
        'NUM_WARPS': 4,
        'USE_WMMA': True,  # Enable WMMA instructions
    }

# Autotuning configurations
class AutotuneConfig:
    """Autotuning parameter ranges for V100 optimization with Tensor Cores"""
    
    # V100 Tensor Core optimal block sizes (16의 배수)
    BLOCK_M_OPTIONS = [32, 48, 64, 80, 96, 112, 128]
    BLOCK_N_OPTIONS = [32, 48, 64, 80, 96, 112, 128] 
    BLOCK_K_OPTIONS = [32, 48, 64, 80, 96, 112, 128]
    NUM_STAGES_OPTIONS = [2, 3, 4, 5]
    NUM_WARPS_OPTIONS = [4, 8]
    
    # Sequence length thresholds for config selection
    SMALL_SEQ_THRESHOLD = 512
    MEDIUM_SEQ_THRESHOLD = 2048
    LARGE_SEQ_THRESHOLD = 8192

# Numerical constants
class NumericalConfig:
    """Numerical stability and precision settings"""
    
    # Softmax numerical stability
    SOFTMAX_EPSILON = 1e-6
    SOFTMAX_SCALE_THRESHOLD = 1e4
    
    # Gradient clipping
    GRAD_CLIP_VALUE = 1.0
    
    # Attention scale default (1/sqrt(head_dim))
    DEFAULT_SCALE_POWER = -0.5

# Performance optimization settings
class OptimizationConfig:
    """Performance optimization flags and settings"""
    
    # Enable/disable optimizations
    ENABLE_AUTOTUNING = True
    ENABLE_KERNEL_CACHING = True
    ENABLE_MEMORY_POOLING = True
    ENABLE_TENSOR_CORES = True  # V100 has 1st gen Tensor Cores for FP16!
    
    # Kernel compilation settings
    TRITON_COMPILE_WARMUP = 3
    TRITON_CACHE_DIR = "/tmp/triton_cache"
    
    # Memory management
    TORCH_MEMORY_FRACTION = 0.9
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True

# Debugging and profiling
class DebugConfig:
    """Debug and profiling configuration"""
    
    ENABLE_PROFILING = False
    ENABLE_KERNEL_DEBUG = False
    LOG_MEMORY_USAGE = False
    VALIDATE_OUTPUTS = False

# Default configurations based on sequence length
def get_default_config(seq_len: int, head_dim: int) -> dict:
    """
    Get default configuration based on sequence length and head dimension
    
    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        
    Returns:
        Dictionary with optimal configuration parameters
    """
    if seq_len <= AutotuneConfig.SMALL_SEQ_THRESHOLD:
        base_config = BlockConfig.SMALL_SEQ_CONFIG.copy()
    elif seq_len <= AutotuneConfig.MEDIUM_SEQ_THRESHOLD:
        base_config = BlockConfig.MEDIUM_SEQ_CONFIG.copy()
    else:
        base_config = BlockConfig.LARGE_SEQ_CONFIG.copy()
    
    # Adjust based on head dimension
    if head_dim <= 64:
        base_config['BLOCK_K'] = min(base_config['BLOCK_K'], 64)
    elif head_dim <= 128:
        base_config['BLOCK_K'] = min(base_config['BLOCK_K'], 128)
    
    return base_config

# Export configuration classes
__all__ = [
    'V100MemoryHierarchy',
    'BlockConfig', 
    'AutotuneConfig',
    'NumericalConfig',
    'OptimizationConfig',
    'DebugConfig',
    'SUPPORTED_DTYPES',
    'V100_COMPUTE_CAPABILITY',
    'MIN_CUDA_VERSION',
    'get_default_config'
]