# config.py
"""
Configuration and constants for Flash Attention V100
Enhanced with advanced V100 optimizations and multi-GPU support
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

# V100 specific memory hierarchy (enhanced with detailed specs)
class V100MemoryHierarchy:
    """V100 GPU memory hierarchy specifications with optimization parameters"""
    
    # SRAM (Shared Memory) specifications
    SRAM_BANDWIDTH = 19 * 1024**4  # 19 TB/s in bytes/s
    SRAM_SIZE = 96 * 1024  # 96KB shared memory per SM
    SRAM_BANKS = 32  # Number of shared memory banks for bank conflict avoidance
    
    # HBM (High Bandwidth Memory) specifications  
    HBM_BANDWIDTH = 900 * 1024**3  # ~900 GB/s for V100
    HBM_SIZE = 16 * 1024**3  # 16GB for V100
    HBM_MEMORY_CONTROLLERS = 4  # Number of memory controllers
    
    # Cache specifications
    L1_CACHE_SIZE = 128 * 1024  # 128KB L1 cache per SM
    L2_CACHE_SIZE = 6 * 1024 * 1024  # 6MB L2 cache
    
    # Warp and thread specifications
    MAX_WARPS_PER_SM = 64
    MAX_THREADS_PER_SM = 2048
    WARP_SIZE = 32
    
    # Memory coalescing specifications
    MEMORY_TRANSACTION_SIZE = 128  # 128-byte transactions
    OPTIMAL_ALIGNMENT = 256  # 256-byte alignment for best performance

# Enhanced block size configurations for V100 with advanced optimizations
class BlockConfig:
    """Block size configurations optimized for V100 with Tensor Cores and memory coalescing"""
    
    # Default block sizes (optimized for V100 Tensor Cores - 16의 배수)
    DEFAULT_BLOCK_M = 64
    DEFAULT_BLOCK_N = 64
    DEFAULT_BLOCK_K = 64
    
    # Warp specialization configurations
    WARP_SPECIALIZATION_CONFIG = {
        'compute_warps': 2,  # Warps dedicated to computation
        'memory_warps': 2,   # Warps dedicated to memory operations
        'total_warps': 4,
    }
    
    # Pipeline stage configurations for different scenarios
    PIPELINE_CONFIGS = {
        'small_seq': {
            'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32,
            'NUM_STAGES': 3,  # Increased for better pipelining
            'NUM_WARPS': 4,
            'ENABLE_WARP_SPECIALIZATION': True,
            'PREFETCH_SIZE': 2,
        },
        'medium_seq': {
            'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64,
            'NUM_STAGES': 4,  # Enhanced pipelining
            'NUM_WARPS': 6,   # More warps for better overlap
            'ENABLE_WARP_SPECIALIZATION': True,
            'PREFETCH_SIZE': 3,
        },
        'large_seq': {
            'BLOCK_M': 96, 'BLOCK_N': 64, 'BLOCK_K': 64,
            'NUM_STAGES': 5,  # Maximum pipelining
            'NUM_WARPS': 8,
            'ENABLE_WARP_SPECIALIZATION': True,
            'PREFETCH_SIZE': 4,
        },
        'xlarge_seq': {
            'BLOCK_M': 128, 'BLOCK_N': 96, 'BLOCK_K': 64,
            'NUM_STAGES': 6,
            'NUM_WARPS': 8,
            'ENABLE_WARP_SPECIALIZATION': True,
            'PREFETCH_SIZE': 4,
        }
    }
    
    # Memory access optimization configurations
    MEMORY_ACCESS_CONFIG = {
        'vectorized_load_size': 128,  # 128-bit vectorized loads
        'bank_conflict_avoidance': True,
        'coalescing_alignment': 256,
        'double_buffering': True,
    }

# Enhanced autotuning configurations with dynamic adaptation
class AutotuneConfig:
    """Enhanced autotuning parameter ranges for V100 optimization"""
    
    # V100 Tensor Core optimal block sizes (확장된 옵션)
    BLOCK_M_OPTIONS = [32, 48, 64, 80, 96, 112, 128, 144, 160]
    BLOCK_N_OPTIONS = [32, 48, 64, 80, 96, 112, 128, 144, 160] 
    BLOCK_K_OPTIONS = [32, 48, 64, 80, 96, 112, 128]
    NUM_STAGES_OPTIONS = [2, 3, 4, 5, 6]  # Extended for better pipelining
    NUM_WARPS_OPTIONS = [4, 6, 8, 12]  # More warp options
    
    # Dynamic tuning parameters
    SEQUENCE_LENGTH_THRESHOLDS = {
        'tiny': 128,
        'small': 512,
        'medium': 2048,
        'large': 8192,
        'xlarge': 32768,
    }
    
    # Performance-based autotuning
    AUTOTUNING_STRATEGIES = {
        'speed_focused': {
            'prefer_large_blocks': True,
            'max_stages': 6,
            'enable_specialization': True,
        },
        'memory_focused': {
            'prefer_small_blocks': True,
            'max_stages': 3,
            'enable_specialization': False,
        },
        'balanced': {
            'prefer_large_blocks': False,
            'max_stages': 4,
            'enable_specialization': True,
        }
    }

# Multi-GPU configuration
class MultiGPUConfig:
    """Multi-GPU optimization settings"""
    
    # Parallelization strategies
    PARALLELIZATION_STRATEGIES = {
        'head_parallel': {
            'split_dimension': 'num_heads',
            'communication_backend': 'nccl',
            'overlap_computation': True,
        },
        'sequence_parallel': {
            'split_dimension': 'sequence_length',
            'communication_backend': 'nccl', 
            'overlap_computation': True,
        },
        'batch_parallel': {
            'split_dimension': 'batch_size',
            'communication_backend': 'nccl',
            'overlap_computation': False,
        }
    }
    
    # Communication optimization
    COMMUNICATION_CONFIG = {
        'enable_gradient_compression': True,
        'compression_ratio': 0.1,
        'async_communication': True,
        'pipeline_parallel_microbatch_size': 4,
    }

# Enhanced numerical constants for stability
class NumericalConfig:
    """Enhanced numerical stability and precision settings"""
    
    # Softmax numerical stability (improved)
    SOFTMAX_EPSILON = 1e-8  # Tighter epsilon
    SOFTMAX_SCALE_THRESHOLD = 1e4
    DYNAMIC_SCALING = True  # Enable dynamic loss scaling
    
    # FP16 specific optimizations
    FP16_CONFIG = {
        'loss_scale': 65536,  # Initial loss scale
        'loss_scale_window': 2000,
        'min_loss_scale': 1.0,
        'gradient_clipping': True,
        'clip_value': 1.0,
    }
    
    # Mixed precision settings
    MIXED_PRECISION_CONFIG = {
        'enable_autocast': True,
        'autocast_dtype': torch.float16,
        'backward_passes_per_step': 1,
        'scaler_growth_factor': 2.0,
        'scaler_backoff_factor': 0.5,
    }
    
    # Attention scale default (1/sqrt(head_dim))
    DEFAULT_SCALE_POWER = -0.5
    ATTENTION_TEMPERATURE = 1.0

# Enhanced performance optimization settings
class OptimizationConfig:
    """Enhanced performance optimization flags and settings"""
    
    # Core optimizations
    ENABLE_AUTOTUNING = True
    ENABLE_KERNEL_CACHING = True
    ENABLE_MEMORY_POOLING = True
    ENABLE_TENSOR_CORES = True
    
    # Advanced optimizations
    ENABLE_WARP_SPECIALIZATION = True
    ENABLE_DOUBLE_BUFFERING = True
    ENABLE_PIPELINE_PARALLELISM = True
    ENABLE_KERNEL_FUSION = True
    ENABLE_ASYNC_MEMORY_COPY = True
    
    # Multi-GPU optimizations
    ENABLE_MULTI_GPU = True
    ENABLE_GRADIENT_CHECKPOINTING = False  # Default off for speed
    ENABLE_ACTIVATION_CHECKPOINTING = False
    
    # Kernel compilation settings (enhanced)
    TRITON_COMPILE_WARMUP = 5  # Increased warmup
    TRITON_CACHE_DIR = "/tmp/triton_cache_v100"
    TRITON_MAX_COMPILATION_TIME = 300  # 5 minutes max
    
    # Memory management (enhanced)
    TORCH_MEMORY_FRACTION = 0.85  # Slightly reduced for stability
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    MEMORY_POOL_SIZE_GB = 4.0  # Larger pool for V100
    
    # Performance monitoring
    ENABLE_PERFORMANCE_PROFILING = False
    PROFILE_MEMORY_USAGE = False
    BENCHMARK_WARMUP_ITERATIONS = 10
    BENCHMARK_TIMING_ITERATIONS = 50

# Kernel fusion configurations
class KernelFusionConfig:
    """Configuration for fused operations"""
    
    # Dropout fusion
    FUSED_DROPOUT_CONFIG = {
        'enable_dropout_fusion': True,
        'dropout_probability': 0.1,
        'use_fast_dropout': True,  # V100 optimized dropout
    }
    
    # Backward pass fusion
    FUSED_BACKWARD_CONFIG = {
        'enable_single_backward_kernel': True,
        'fuse_delta_computation': True,
        'enable_gradient_accumulation': True,
    }
    
    # Softmax fusion options
    FUSED_SOFTMAX_CONFIG = {
        'fuse_scale_mask_softmax': True,
        'enable_fast_softmax': True,
        'use_tensor_cores_for_softmax': False,  # Usually not beneficial
    }

# Enhanced debugging and profiling
class DebugConfig:
    """Enhanced debug and profiling configuration"""
    
    ENABLE_PROFILING = False
    ENABLE_KERNEL_DEBUG = False
    LOG_MEMORY_USAGE = False
    VALIDATE_OUTPUTS = False
    
    # Advanced debugging
    ENABLE_KERNEL_TIMING = False
    ENABLE_MEMORY_TRACKING = False
    SAVE_INTERMEDIATE_OUTPUTS = False
    CHECK_NUMERICAL_STABILITY = False
    
    # Performance analysis
    PROFILE_WARP_EFFICIENCY = False
    PROFILE_MEMORY_BANDWIDTH = False
    PROFILE_TENSOR_CORE_UTILIZATION = False

# Enhanced configuration selection based on workload
def get_enhanced_config(seq_len: int, head_dim: int, num_gpus: int = 1, 
                       strategy: str = 'balanced') -> dict:
    """
    Get enhanced configuration based on sequence length, head dimension, and hardware
    
    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        num_gpus: Number of GPUs available
        strategy: Optimization strategy ('speed_focused', 'memory_focused', 'balanced')
        
    Returns:
        Dictionary with optimal configuration parameters
    """
    
    # Determine sequence category
    if seq_len <= AutotuneConfig.SEQUENCE_LENGTH_THRESHOLDS['tiny']:
        base_config = BlockConfig.PIPELINE_CONFIGS['small_seq'].copy()
    elif seq_len <= AutotuneConfig.SEQUENCE_LENGTH_THRESHOLDS['small']:
        base_config = BlockConfig.PIPELINE_CONFIGS['small_seq'].copy()
    elif seq_len <= AutotuneConfig.SEQUENCE_LENGTH_THRESHOLDS['medium']:
        base_config = BlockConfig.PIPELINE_CONFIGS['medium_seq'].copy()
    elif seq_len <= AutotuneConfig.SEQUENCE_LENGTH_THRESHOLDS['large']:
        base_config = BlockConfig.PIPELINE_CONFIGS['large_seq'].copy()
    else:
        base_config = BlockConfig.PIPELINE_CONFIGS['xlarge_seq'].copy()
    
    # Apply strategy-specific adjustments
    strategy_config = AutotuneConfig.AUTOTUNING_STRATEGIES.get(strategy, 
                      AutotuneConfig.AUTOTUNING_STRATEGIES['balanced'])
    
    if strategy_config['prefer_large_blocks'] and seq_len > 1024:
        base_config['BLOCK_M'] = min(base_config['BLOCK_M'] * 2, 160)
        base_config['BLOCK_N'] = min(base_config['BLOCK_N'] * 2, 160)
    
    # Adjust for head dimension
    if head_dim <= 32:
        base_config['BLOCK_M'] = min(base_config['BLOCK_M'], 96)
        base_config['BLOCK_N'] = min(base_config['BLOCK_N'], 96)
    elif head_dim >= 128:
        base_config['NUM_WARPS'] = min(base_config['NUM_WARPS'], 6)
    
    # Multi-GPU adjustments
    if num_gpus > 1:
        base_config['ENABLE_MULTI_GPU'] = True
        base_config['NUM_GPUS'] = num_gpus
        # Prefer head parallelization for multi-GPU
        base_config['PARALLELIZATION_STRATEGY'] = 'head_parallel'
    
    # Always set BLOCK_K to head_dim for optimal memory access
    base_config['BLOCK_K'] = head_dim
    
    # Add optimization flags
    base_config.update({
        'ENABLE_WARP_SPECIALIZATION': strategy_config['enable_specialization'],
        'ENABLE_DOUBLE_BUFFERING': True,
        'ENABLE_KERNEL_FUSION': True,
        'MEMORY_ACCESS_OPTIMIZATION': True,
    })
    
    return base_config

# Performance monitoring utilities
class PerformanceConfig:
    """Performance monitoring and optimization configuration"""
    
    # Target performance metrics for V100
    TARGET_METRICS = {
        'peak_tflops': 125,  # V100 theoretical peak for FP16
        'target_efficiency': 0.65,  # 65% efficiency target
        'memory_bandwidth_utilization': 0.80,  # 80% HBM utilization
        'tensor_core_utilization': 0.75,  # 75% Tensor Core utilization
    }
    
    # Benchmark configurations
    BENCHMARK_CONFIG = {
        'warmup_iterations': 10,
        'timing_iterations': 50,
        'memory_measurement_interval': 5,
        'enable_detailed_profiling': False,
    }

# Export all configuration classes
__all__ = [
    'V100MemoryHierarchy',
    'BlockConfig', 
    'AutotuneConfig',
    'MultiGPUConfig',
    'NumericalConfig',
    'OptimizationConfig',
    'KernelFusionConfig',
    'DebugConfig',
    'PerformanceConfig',
    'SUPPORTED_DTYPES',
    'V100_COMPUTE_CAPABILITY',
    'MIN_CUDA_VERSION',
    'get_enhanced_config'
]