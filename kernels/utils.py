# utils.py
"""
Utility functions for Flash Attention V100 kernels

This module provides helper functions for kernel management,
validation, and metadata extraction.
"""

import torch
import triton
from typing import Dict, Any, Tuple, Optional
import math

from ..config import V100MemoryHierarchy, BlockConfig


def get_kernel_metadata(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_warps: int,
    head_dim: int,
    dtype: torch.dtype
) -> Dict[str, Any]:
    """
    Get metadata about kernel configuration
    
    Args:
        block_m: Block size for M dimension
        block_n: Block size for N dimension  
        block_k: Block size for K dimension
        num_stages: Number of pipeline stages
        num_warps: Number of warps
        head_dim: Head dimension
        dtype: Data type
        
    Returns:
        Dictionary with kernel metadata
    """
    
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Calculate memory usage
    memory_usage = estimate_kernel_memory_usage(
        block_m, block_n, block_k, head_dim, dtype
    )
    
    # Calculate theoretical occupancy
    occupancy = calculate_theoretical_occupancy(
        memory_usage['shared_memory_bytes'], num_warps
    )
    
    # Calculate memory bandwidth requirements
    memory_bandwidth = calculate_memory_bandwidth_requirements(
        block_m, block_n, head_dim, element_size
    )
    
    return {
        'block_sizes': {
            'BLOCK_M': block_m,
            'BLOCK_N': block_n,
            'BLOCK_K': block_k,
        },
        'pipeline_config': {
            'num_stages': num_stages,
            'num_warps': num_warps,
        },
        'memory_usage': memory_usage,
        'theoretical_occupancy': occupancy,
        'memory_bandwidth_gbps': memory_bandwidth,
        'compatibility': {
            'v100_compatible': is_v100_compatible(memory_usage, num_warps),
            'memory_fits': memory_usage['shared_memory_bytes'] <= V100MemoryHierarchy.SRAM_SIZE,
        }
    }


def estimate_kernel_memory_usage(
    block_m: int,
    block_n: int, 
    block_k: int,
    head_dim: int,
    dtype: torch.dtype
) -> Dict[str, int]:
    """
    Estimate memory usage for Flash Attention kernel
    
    Args:
        block_m, block_n, block_k: Block dimensions
        head_dim: Head dimension
        dtype: Data type
        
    Returns:
        Dictionary with memory usage breakdown
    """
    
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Q block: [BLOCK_M, head_dim]
    q_block_size = block_m * head_dim * element_size
    
    # K block: [BLOCK_N, head_dim] 
    k_block_size = block_n * head_dim * element_size
    
    # V block: [BLOCK_N, head_dim]
    v_block_size = block_n * head_dim * element_size
    
    # Attention scores: [BLOCK_M, BLOCK_N] (usually fp32 for numerical stability)
    attention_scores_size = block_m * block_n * 4  # fp32
    
    # Output accumulator: [BLOCK_M, head_dim] (usually fp32)
    output_acc_size = block_m * head_dim * 4  # fp32
    
    # Statistics arrays (LSE, max): [BLOCK_M] each (fp32)
    statistics_size = 2 * block_m * 4  # fp32
    
    # Total shared memory usage
    shared_memory_bytes = (
        q_block_size + k_block_size + v_block_size + 
        attention_scores_size + output_acc_size + statistics_size
    )
    
    # Add some overhead for intermediate computations
    overhead = int(shared_memory_bytes * 0.1)  # 10% overhead
    shared_memory_bytes += overhead
    
    return {
        'q_block_bytes': q_block_size,
        'k_block_bytes': k_block_size,
        'v_block_bytes': v_block_size,
        'attention_scores_bytes': attention_scores_size,
        'output_accumulator_bytes': output_acc_size,
        'statistics_bytes': statistics_size,
        'overhead_bytes': overhead,
        'shared_memory_bytes': shared_memory_bytes,
        'shared_memory_kb': shared_memory_bytes / 1024,
    }


def calculate_theoretical_occupancy(
    shared_memory_bytes: int,
    num_warps: int
) -> Dict[str, float]:
    """
    Calculate theoretical GPU occupancy
    
    Args:
        shared_memory_bytes: Shared memory usage per block
        num_warps: Number of warps per block
        
    Returns:
        Dictionary with occupancy metrics
    """
    
    # V100 specifications
    v100_specs = {
        'max_threads_per_sm': 2048,
        'max_blocks_per_sm': 32,
        'max_warps_per_sm': 64,
        'shared_memory_per_sm': V100MemoryHierarchy.SRAM_SIZE,
        'threads_per_warp': 32,
    }
    
    threads_per_block = num_warps * v100_specs['threads_per_warp']
    
    # Occupancy limitations
    occupancy_by_threads = v100_specs['max_threads_per_sm'] / threads_per_block
    occupancy_by_blocks = v100_specs['max_blocks_per_sm']
    occupancy_by_warps = v100_specs['max_warps_per_sm'] / num_warps
    occupancy_by_sram = v100_specs['shared_memory_per_sm'] / shared_memory_bytes if shared_memory_bytes > 0 else float('inf')
    
    # Theoretical occupancy is limited by the most restrictive factor
    theoretical_occupancy = min(
        occupancy_by_threads,
        occupancy_by_blocks,
        occupancy_by_warps,
        occupancy_by_sram
    )
    
    # Normalize to percentage
    max_possible_blocks = min(
        v100_specs['max_blocks_per_sm'],
        v100_specs['max_threads_per_sm'] // threads_per_block,
        v100_specs['max_warps_per_sm'] // num_warps
    )
    
    occupancy_percentage = (theoretical_occupancy / max_possible_blocks) * 100 if max_possible_blocks > 0 else 0
    
    return {
        'theoretical_blocks_per_sm': theoretical_occupancy,
        'occupancy_percentage': occupancy_percentage,
        'limiting_factor': get_limiting_factor(
            occupancy_by_threads, occupancy_by_blocks, 
            occupancy_by_warps, occupancy_by_sram
        ),
        'threads_per_block': threads_per_block,
    }


def get_limiting_factor(occ_threads: float, occ_blocks: float, occ_warps: float, occ_sram: float) -> str:
    """Determine which factor limits occupancy"""
    min_occupancy = min(occ_threads, occ_blocks, occ_warps, occ_sram)
    
    if min_occupancy == occ_threads:
        return "threads_per_sm"
    elif min_occupancy == occ_blocks:
        return "blocks_per_sm"
    elif min_occupancy == occ_warps:
        return "warps_per_sm"
    else:
        return "shared_memory"


def calculate_memory_bandwidth_requirements(
    block_m: int,
    block_n: int,
    head_dim: int,
    element_size: int
) -> float:
    """
    Calculate memory bandwidth requirements for kernel
    
    Args:
        block_m, block_n: Block dimensions
        head_dim: Head dimension
        element_size: Size of each element in bytes
        
    Returns:
        Required memory bandwidth in GB/s
    """
    
    # Data loaded per block iteration
    q_load = block_m * head_dim * element_size
    k_load = block_n * head_dim * element_size  
    v_load = block_n * head_dim * element_size
    
    # Data stored per block iteration
    output_store = block_m * head_dim * element_size
    
    # Total data movement per block
    total_bytes_per_block = q_load + k_load + v_load + output_store
    
    # Assume 1ms per block iteration (rough estimate)
    estimated_time_per_block_ms = 1.0
    
    # Required bandwidth
    bandwidth_gbps = (total_bytes_per_block / (estimated_time_per_block_ms / 1000)) / 1e9
    
    return bandwidth_gbps


def is_v100_compatible(memory_usage: Dict[str, int], num_warps: int) -> bool:
    """
    Check if configuration is compatible with V100
    
    Args:
        memory_usage: Memory usage breakdown
        num_warps: Number of warps
        
    Returns:
        True if compatible with V100
    """
    
    # Check shared memory limit
    if memory_usage['shared_memory_bytes'] > V100MemoryHierarchy.SRAM_SIZE:
        return False
    
    # Check warp limit (V100 has 64 warps per SM)
    if num_warps > 64:
        return False
    
    # Check reasonable occupancy
    occupancy = calculate_theoretical_occupancy(
        memory_usage['shared_memory_bytes'], num_warps
    )
    
    if occupancy['occupancy_percentage'] < 10:  # Less than 10% occupancy is usually bad
        return False
    
    return True


def validate_kernel_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_warps: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    dtype: torch.dtype
) -> Tuple[bool, Optional[str]]:
    """
    Validate kernel configuration parameters
    
    Args:
        block_m, block_n, block_k: Block dimensions
        num_stages: Number of pipeline stages
        num_warps: Number of warps
        seq_len_q, seq_len_k: Sequence lengths
        head_dim: Head dimension
        dtype: Data type
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    # Check block sizes are positive powers of 2
    if not all(is_power_of_2(x) and x > 0 for x in [block_m, block_n]):
        return False, "Block sizes must be positive powers of 2"
    
    # Check block_k matches head_dim (required for correctness)
    if block_k != head_dim:
        return False, f"BLOCK_K ({block_k}) must equal head_dim ({head_dim})"
    
    # Check block sizes don't exceed sequence lengths
    if block_m > seq_len_q:
        return False, f"BLOCK_M ({block_m}) cannot exceed seq_len_q ({seq_len_q})"
    
    if block_n > seq_len_k:
        return False, f"BLOCK_N ({block_n}) cannot exceed seq_len_k ({seq_len_k})"
    
    # Check number of warps is valid
    if num_warps not in [1, 2, 4, 8, 16, 32]:
        return False, f"num_warps ({num_warps}) must be a power of 2 between 1 and 32"
    
    # Check number of stages is reasonable
    if num_stages < 1 or num_stages > 8:
        return False, f"num_stages ({num_stages}) must be between 1 and 8"
    
    # Check memory requirements
    memory_usage = estimate_kernel_memory_usage(block_m, block_n, block_k, head_dim, dtype)
    
    if memory_usage['shared_memory_bytes'] > V100MemoryHierarchy.SRAM_SIZE:
        return False, f"Shared memory usage ({memory_usage['shared_memory_kb']:.1f} KB) exceeds V100 limit"
    
    # Check V100 compatibility
    if not is_v100_compatible(memory_usage, num_warps):
        return False, "Configuration not compatible with V100 architecture"
    
    return True, None


def is_power_of_2(n: int) -> bool:
    """Check if number is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def round_up_to_power_of_2(n: int) -> int:
    """Round up to the nearest power of 2"""
    return 2 ** math.ceil(math.log2(n))


def get_optimal_block_size_for_sequence(seq_len: int, max_block_size: int = 128) -> int:
    """
    Get optimal block size for given sequence length
    
    Args:
        seq_len: Sequence length
        max_block_size: Maximum allowed block size
        
    Returns:
        Optimal block size (power of 2)
    """
    
    # Find largest power of 2 that divides seq_len evenly
    optimal_size = max_block_size
    
    while optimal_size > 16:  # Minimum reasonable block size
        if seq_len % optimal_size == 0:
            return optimal_size
        optimal_size //= 2
    
    # If no even divisor found, use the largest power of 2 <= seq_len
    return min(round_up_to_power_of_2(seq_len // 4), max_block_size)


# Export utility functions
__all__ = [
    'get_kernel_metadata',
    'estimate_kernel_memory_usage',
    'calculate_theoretical_occupancy',
    'calculate_memory_bandwidth_requirements',
    'is_v100_compatible',
    'validate_kernel_config',
    'is_power_of_2',
    'round_up_to_power_of_2',
    'get_optimal_block_size_for_sequence',
]