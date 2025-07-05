# autotuning.py
"""
Autotuning utilities for Flash Attention V100 kernels

This module provides automatic kernel configuration optimization
specifically tuned for V100 architecture.
"""

import torch
import triton
import time
import itertools
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

from ..config import AutotuneConfig, V100MemoryHierarchy, OptimizationConfig
from ..utils import cdiv


@dataclass
class KernelConfig:
    """Kernel configuration for Flash Attention"""
    BLOCK_M: int
    BLOCK_N: int 
    BLOCK_K: int
    NUM_STAGES: int
    NUM_WARPS: int
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for Triton config"""
        return {
            'BLOCK_M': self.BLOCK_M,
            'BLOCK_N': self.BLOCK_N,
            'BLOCK_K': self.BLOCK_K,
        }
    
    def __str__(self) -> str:
        return f"BLOCK_M={self.BLOCK_M}, BLOCK_N={self.BLOCK_N}, BLOCK_K={self.BLOCK_K}, NUM_STAGES={self.NUM_STAGES}, NUM_WARPS={self.NUM_WARPS}"


@dataclass
class BenchmarkResult:
    """Result of kernel benchmarking"""
    config: KernelConfig
    avg_time_ms: float
    throughput_tflops: float
    memory_bandwidth_gbps: float
    success: bool
    error_msg: Optional[str] = None


class AutoTuner:
    """
    Automatic kernel configuration tuner for V100
    
    This class performs empirical search over kernel configurations
    to find the optimal settings for given workload characteristics.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize AutoTuner
        
        Args:
            cache_dir: Directory to cache tuning results
        """
        self.cache_dir = cache_dir or OptimizationConfig.TRITON_CACHE_DIR
        self.cache_file = os.path.join(self.cache_dir, "autotuning_cache.json")
        self.cache = self._load_cache()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached tuning results"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load autotuning cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save tuning results to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save autotuning cache: {e}")
    
    def _get_cache_key(self, batch_size: int, seq_len_q: int, seq_len_k: int, 
                      num_heads: int, head_dim: int, dtype: str, causal: bool) -> str:
        """Generate cache key for given workload"""
        return f"{batch_size}_{seq_len_q}_{seq_len_k}_{num_heads}_{head_dim}_{dtype}_{causal}"
    
    def get_candidate_configs(self, seq_len_q: int, seq_len_k: int, head_dim: int) -> List[KernelConfig]:
        """
        Generate candidate configurations based on problem size
        
        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key/Value sequence length
            head_dim: Head dimension
            
        Returns:
            List of candidate kernel configurations
        """
        configs = []
        
        # Base configurations from AutotuneConfig
        block_m_options = AutotuneConfig.BLOCK_M_OPTIONS
        block_n_options = AutotuneConfig.BLOCK_N_OPTIONS
        block_k_options = [head_dim]  # Block K typically matches head_dim
        stages_options = AutotuneConfig.NUM_STAGES_OPTIONS
        warps_options = AutotuneConfig.NUM_WARPS_OPTIONS
        
        # Filter configurations based on problem size and V100 Tensor Core constraints
        for block_m, block_n, block_k, stages, warps in itertools.product(
            block_m_options, block_n_options, block_k_options, stages_options, warps_options
        ):
            # Skip if block sizes are larger than sequence lengths
            if block_m > seq_len_q or block_n > seq_len_k:
                continue
            
            # V100 Tensor Core optimization: prefer 16의 배수 for FP16 operations
            if block_m % 16 != 0 or block_n % 16 != 0:
                continue
            
            # Check memory constraints (shared memory usage)
            estimated_sram_usage = self._estimate_sram_usage(block_m, block_n, head_dim)
            if estimated_sram_usage > V100MemoryHierarchy.SRAM_SIZE:
                continue
            
            # V100 specific constraints
            if warps > 8:  # V100 has limited warp scheduling
                continue
            
            if stages > 5:  # Too many stages can hurt V100 performance
                continue
            
            configs.append(KernelConfig(block_m, block_n, block_k, stages, warps))
        
        return configs
    
    def _estimate_sram_usage(self, block_m: int, block_n: int, head_dim: int) -> int:
        """Estimate shared memory usage for given block configuration"""
        # Rough estimation: Q block + K block + V block + intermediate results
        element_size = 2  # Assume fp16
        
        q_block_size = block_m * head_dim * element_size
        k_block_size = block_n * head_dim * element_size
        v_block_size = block_n * head_dim * element_size
        attention_block_size = block_m * block_n * element_size
        
        total_size = q_block_size + k_block_size + v_block_size + attention_block_size
        return int(total_size * 1.2)  # Add 20% safety margin
    
    def benchmark_config(
        self, 
        config: KernelConfig,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float,
        causal: bool,
        num_warmup: int = 5,
        num_runs: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark a specific kernel configuration
        
        Args:
            config: Kernel configuration to benchmark
            q, k, v: Input tensors
            scale: Attention scale
            causal: Whether to use causal masking
            num_warmup: Number of warmup runs
            num_runs: Number of timed runs
            
        Returns:
            Benchmark result
        """
        from .forward_kernel import flash_attention_forward_triton
        
        try:
            # Warmup runs
            for _ in range(num_warmup):
                _ = flash_attention_forward_triton(
                    q, k, v, scale, causal, 
                    config.BLOCK_M, config.BLOCK_N
                )
            
            torch.cuda.synchronize()
            
            # Timed runs
            start_time = time.time()
            for _ in range(num_runs):
                output, _, _ = flash_attention_forward_triton(
                    q, k, v, scale, causal,
                    config.BLOCK_M, config.BLOCK_N
                )
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / num_runs
            
            # Calculate performance metrics
            batch_size, seq_len_q, num_heads, head_dim = q.shape
            _, seq_len_k, _, _ = k.shape
            
            # Approximate FLOPS (4 * B * H * N * N * D for attention)
            flops = 4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
            throughput_tflops = (flops / (avg_time_ms / 1000)) / 1e12
            
            # Calculate memory bandwidth
            total_memory_bytes = (q.numel() + k.numel() + v.numel() + output.numel()) * q.element_size()
            memory_bandwidth_gbps = (total_memory_bytes / (avg_time_ms / 1000)) / 1e9
            
            return BenchmarkResult(
                config=config,
                avg_time_ms=avg_time_ms,
                throughput_tflops=throughput_tflops,
                memory_bandwidth_gbps=memory_bandwidth_gbps,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                config=config,
                avg_time_ms=float('inf'),
                throughput_tflops=0.0,
                memory_bandwidth_gbps=0.0,
                success=False,
                error_msg=str(e)
            )
    
    def tune(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        causal: bool = False,
        use_cache: bool = True
    ) -> KernelConfig:
        """
        Find optimal kernel configuration for given inputs
        
        Args:
            q, k, v: Input tensors
            scale: Attention scale
            causal: Whether to use causal masking
            use_cache: Whether to use cached results
            
        Returns:
            Optimal kernel configuration
        """
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        cache_key = self._get_cache_key(
            batch_size, seq_len_q, seq_len_k, num_heads, head_dim, str(q.dtype), causal
        )
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cached_config = self.cache[cache_key]
            return KernelConfig(**cached_config)
        
        # Generate candidate configurations
        candidates = self.get_candidate_configs(seq_len_q, seq_len_k, head_dim)
        
        if not candidates:
            # Fallback to default configuration
            from ..config import BlockConfig
            return KernelConfig(
                BLOCK_M=BlockConfig.DEFAULT_BLOCK_M,
                BLOCK_N=BlockConfig.DEFAULT_BLOCK_N, 
                BLOCK_K=head_dim,
                NUM_STAGES=3,
                NUM_WARPS=4
            )
        
        print(f"Autotuning Flash Attention for shape {q.shape} with {len(candidates)} configurations...")
        
        # Benchmark all candidates
        results = []
        for i, config in enumerate(candidates):
            if i % 5 == 0:
                print(f"  Progress: {i+1}/{len(candidates)}")
            
            result = self.benchmark_config(q, k, v, scale, causal, config)
            results.append(result)
        
        # Find best configuration (minimize time)
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            print("Warning: No configurations succeeded, using default")
            from ..config import BlockConfig
            return KernelConfig(
                BLOCK_M=BlockConfig.DEFAULT_BLOCK_M,
                BLOCK_N=BlockConfig.DEFAULT_BLOCK_N,
                BLOCK_K=head_dim,
                NUM_STAGES=3,
                NUM_WARPS=4
            )
        
        best_result = min(successful_results, key=lambda r: r.avg_time_ms)
        best_config = best_result.config
        
        print(f"Best configuration: {best_config}")
        print(f"  Time: {best_result.avg_time_ms:.2f} ms")
        print(f"  Throughput: {best_result.throughput_tflops:.2f} TFLOPS")
        print(f"  Memory BW: {best_result.memory_bandwidth_gbps:.1f} GB/s")
        
        # Cache the result
        if use_cache:
            self.cache[cache_key] = best_config.to_dict()
            self.cache[cache_key].update({
                'NUM_STAGES': best_config.NUM_STAGES,
                'NUM_WARPS': best_config.NUM_WARPS
            })
            self._save_cache()
        
        return best_config


def get_autotuning_config(
    seq_len_q: int, 
    seq_len_k: int, 
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """
    Get pre-tuned configuration for common workloads
    
    This function provides reasonable configurations without running
    full autotuning, based on empirical results.
    
    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length  
        head_dim: Head dimension
        dtype: Data type
        
    Returns:
        Dictionary with kernel configuration
    """
    
    # Pre-tuned configurations for common scenarios on V100 with Tensor Cores
    configs = {
        # Small sequences (≤512) - Tensor Core optimized
        'small': {
            'BLOCK_M': 32,  # 16의 배수
            'BLOCK_N': 32,  # 16의 배수
            'NUM_STAGES': 2,
            'NUM_WARPS': 4,
        },
        # Medium sequences (513-2048) - Tensor Core optimized
        'medium': {
            'BLOCK_M': 64,  # 16의 배수
            'BLOCK_N': 64,  # 16의 배수
            'NUM_STAGES': 3,
            'NUM_WARPS': 4,
        },
        # Large sequences (2049-8192) - Tensor Core optimized
        'large': {
            'BLOCK_M': 96,  # 16의 배수
            'BLOCK_N': 64,  # 16의 배수
            'NUM_STAGES': 4,
            'NUM_WARPS': 8,
        },
        # Very large sequences (>8192) - Tensor Core optimized
        'xlarge': {
            'BLOCK_M': 128,  # 16의 배수
            'BLOCK_N': 96,   # 16의 배수
            'NUM_STAGES': 4,
            'NUM_WARPS': 8,
        }
    }
    
    max_seq_len = max(seq_len_q, seq_len_k)
    
    if max_seq_len <= 512:
        config = configs['small'].copy()
    elif max_seq_len <= 2048:
        config = configs['medium'].copy()
    elif max_seq_len <= 8192:
        config = configs['large'].copy()
    else:
        config = configs['xlarge'].copy()
    
    # Always set BLOCK_K to head_dim for optimal memory access
    config['BLOCK_K'] = head_dim
    
    # Adjust for specific head dimensions
    if head_dim <= 32:
        config['BLOCK_M'] = min(config['BLOCK_M'], 64)
        config['BLOCK_N'] = min(config['BLOCK_N'], 64)
    elif head_dim >= 128:
        config['NUM_WARPS'] = min(config['NUM_WARPS'], 4)  # Reduce warps for large head_dim
    
    return config


def benchmark_kernel_configs(
    configs: List[KernelConfig],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False
) -> List[BenchmarkResult]:
    """
    Benchmark multiple kernel configurations
    
    Args:
        configs: List of configurations to benchmark
        q, k, v: Input tensors
        scale: Attention scale
        causal: Whether to use causal masking
        
    Returns:
        List of benchmark results
    """
    tuner = AutoTuner()
    results = []
    
    for config in configs:
        result = tuner.benchmark_config(config, q, k, v, scale, causal)
        results.append(result)
    
    return results


# Export main functions
__all__ = [
    'KernelConfig',
    'BenchmarkResult', 
    'AutoTuner',
    'get_autotuning_config',
    'benchmark_kernel_configs',
]