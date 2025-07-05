# autotuning.py
"""
Enhanced autotuning utilities for Flash Attention V100 kernels

This module provides advanced automatic kernel configuration optimization
with dynamic adaptation, multi-GPU awareness, and performance monitoring.
"""

import torch
import triton
import time
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib

from ..config import AutotuneConfig, V100MemoryHierarchy, OptimizationConfig, PerformanceConfig
from ..utils import cdiv


@dataclass
class EnhancedKernelConfig:
    """Enhanced kernel configuration with additional optimization parameters"""
    BLOCK_M: int
    BLOCK_N: int 
    BLOCK_K: int
    NUM_STAGES: int
    NUM_WARPS: int
    
    # Advanced optimization parameters
    enable_warp_specialization: bool = True
    enable_double_buffering: bool = True
    enable_prefetch: bool = True
    use_fast_math: bool = False
    memory_efficient: bool = True
    
    # Performance hints
    expected_efficiency: float = 0.0
    memory_usage_mb: float = 0.0
    register_pressure: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Triton config"""
        return {
            'BLOCK_M': self.BLOCK_M,
            'BLOCK_N': self.BLOCK_N,
            'BLOCK_K': self.BLOCK_K,
        }
    
    def to_triton_config(self) -> triton.Config:
        """Convert to Triton Config object"""
        return triton.Config(
            self.to_dict(),
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES
        )
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        return f"{self.BLOCK_M}_{self.BLOCK_N}_{self.BLOCK_K}_{self.NUM_STAGES}_{self.NUM_WARPS}"
    
    def __str__(self) -> str:
        return (f"BLOCK_M={self.BLOCK_M}, BLOCK_N={self.BLOCK_N}, BLOCK_K={self.BLOCK_K}, "
                f"NUM_STAGES={self.NUM_STAGES}, NUM_WARPS={self.NUM_WARPS}, "
                f"efficiency={self.expected_efficiency:.2f}")


@dataclass
class EnhancedBenchmarkResult:
    """Enhanced benchmark result with detailed performance metrics"""
    config: EnhancedKernelConfig
    avg_time_ms: float
    std_time_ms: float
    throughput_tflops: float
    memory_bandwidth_gbps: float
    tensor_core_utilization: float
    occupancy_percentage: float
    success: bool
    error_msg: Optional[str] = None
    
    # Additional metrics
    peak_memory_mb: float = 0.0
    compile_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    numerical_accuracy: float = 1.0
    
    def get_performance_score(self, 
                             weight_speed: float = 0.4,
                             weight_memory: float = 0.3,
                             weight_efficiency: float = 0.3) -> float:
        """Calculate overall performance score"""
        if not self.success:
            return 0.0
        
        # Normalize metrics to [0, 1] range
        speed_score = min(self.throughput_tflops / PerformanceConfig.TARGET_METRICS['peak_tflops'], 1.0)
        memory_score = min(self.memory_bandwidth_gbps / (V100MemoryHierarchy.HBM_BANDWIDTH / 1e9), 1.0)
        efficiency_score = min(self.tensor_core_utilization, 1.0)
        
        # Weighted combination
        total_score = (weight_speed * speed_score + 
                      weight_memory * memory_score + 
                      weight_efficiency * efficiency_score)
        
        # Penalty for high memory usage
        memory_penalty = max(0, (self.peak_memory_mb - 8000) / 8000)  # Penalty above 8GB
        total_score *= (1.0 - 0.1 * memory_penalty)
        
        return total_score


class AdvancedAutoTuner:
    """
    Advanced automatic kernel configuration tuner for V100 with intelligent optimization
    
    Features:
    - Dynamic configuration space pruning
    - Multi-objective optimization (speed vs memory)
    - Predictive performance modeling
    - Multi-GPU awareness
    - Adaptive caching with performance trends
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 enable_predictive_modeling: bool = True,
                 enable_multi_gpu: bool = False,
                 optimization_strategy: str = 'balanced'):
        """
        Initialize Advanced AutoTuner
        
        Args:
            cache_dir: Directory to cache tuning results
            enable_predictive_modeling: Whether to use ML-based performance prediction
            enable_multi_gpu: Whether to consider multi-GPU configurations
            optimization_strategy: 'speed', 'memory', 'balanced', or 'adaptive'
        """
        self.cache_dir = cache_dir or OptimizationConfig.TRITON_CACHE_DIR
        self.cache_file = os.path.join(self.cache_dir, "enhanced_autotuning_cache.json")
        self.model_file = os.path.join(self.cache_dir, "performance_model.pkl")
        
        self.enable_predictive_modeling = enable_predictive_modeling
        self.enable_multi_gpu = enable_multi_gpu
        self.optimization_strategy = optimization_strategy
        
        # Cache and performance tracking
        self.cache = self._load_cache()
        self.performance_model = self._load_performance_model()
        self.benchmark_history = []
        
        # Threading for parallel benchmarking
        self.max_workers = min(4, os.cpu_count() or 1)
        
        # Performance targets based on strategy
        self.performance_targets = self._get_performance_targets()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets based on optimization strategy"""
        targets = PerformanceConfig.TARGET_METRICS.copy()
        
        if self.optimization_strategy == 'speed':
            targets.update({
                'target_efficiency': 0.70,
                'memory_weight': 0.2,
                'speed_weight': 0.6,
                'efficiency_weight': 0.2,
            })
        elif self.optimization_strategy == 'memory':
            targets.update({
                'target_efficiency': 0.50,
                'memory_weight': 0.6,
                'speed_weight': 0.2,
                'efficiency_weight': 0.2,
            })
        elif self.optimization_strategy == 'balanced':
            targets.update({
                'target_efficiency': 0.60,
                'memory_weight': 0.33,
                'speed_weight': 0.33,
                'efficiency_weight': 0.34,
            })
        else:  # adaptive
            targets.update({
                'target_efficiency': 0.65,
                'memory_weight': 0.4,
                'speed_weight': 0.4,
                'efficiency_weight': 0.2,
            })
        
        return targets
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached tuning results with validation"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Validate cache format and remove outdated entries
                validated_cache = {}
                for key, value in cache_data.items():
                    if self._validate_cache_entry(value):
                        validated_cache[key] = value
                
                return validated_cache
            except Exception as e:
                print(f"Warning: Could not load autotuning cache: {e}")
        return {}
    
    def _validate_cache_entry(self, entry: Dict) -> bool:
        """Validate cache entry format and freshness"""
        required_keys = ['config', 'performance', 'timestamp', 'device_info']
        return all(key in entry for key in required_keys)
    
    def _save_cache(self):
        """Save tuning results to cache with metadata"""
        try:
            # Add metadata
            cache_metadata = {
                'version': '2.0',
                'strategy': self.optimization_strategy,
                'device_capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
                'last_updated': time.time(),
            }
            
            cache_data = {
                'metadata': cache_metadata,
                'entries': self.cache
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save autotuning cache: {e}")
    
    def _load_performance_model(self):
        """Load ML-based performance prediction model"""
        if not self.enable_predictive_modeling or not os.path.exists(self.model_file):
            return None
        
        try:
            with open(self.model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load performance model: {e}")
            return None
    
    def _save_performance_model(self, model):
        """Save ML-based performance prediction model"""
        if not self.enable_predictive_modeling:
            return
        
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Warning: Could not save performance model: {e}")
    
    def _get_cache_key(self, batch_size: int, seq_len_q: int, seq_len_k: int, 
                      num_heads: int, head_dim: int, dtype: str, causal: bool,
                      num_gpus: int = 1) -> str:
        """Generate comprehensive cache key"""
        device_info = f"v100_{torch.cuda.get_device_capability()}" if torch.cuda.is_available() else "cpu"
        
        key_components = [
            str(batch_size), str(seq_len_q), str(seq_len_k), 
            str(num_heads), str(head_dim), dtype, str(causal),
            str(num_gpus), device_info, self.optimization_strategy
        ]
        
        # Create hash for long keys
        key_string = "_".join(key_components)
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"hash_{key_hash}"
        
        return key_string
    
    def get_intelligent_candidate_configs(self, 
                                        seq_len_q: int, 
                                        seq_len_k: int, 
                                        head_dim: int,
                                        num_gpus: int = 1,
                                        max_configs: int = 20) -> List[EnhancedKernelConfig]:
        """
        Generate intelligent candidate configurations using ML predictions and heuristics
        
        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key/Value sequence length
            head_dim: Head dimension
            num_gpus: Number of GPUs
            max_configs: Maximum number of configurations to generate
            
        Returns:
            List of promising kernel configurations
        """
        configs = []
        
        # Determine sequence category for base configurations
        max_seq_len = max(seq_len_q, seq_len_k)
        
        if max_seq_len <= 512:
            base_configs = self._get_small_sequence_configs(head_dim)
        elif max_seq_len <= 2048:
            base_configs = self._get_medium_sequence_configs(head_dim)
        elif max_seq_len <= 8192:
            base_configs = self._get_large_sequence_configs(head_dim)
        else:
            base_configs = self._get_xlarge_sequence_configs(head_dim)
        
        # Add base configurations
        configs.extend(base_configs)
        
        # Add strategy-specific configurations
        if self.optimization_strategy == 'speed':
            configs.extend(self._get_speed_optimized_configs(seq_len_q, seq_len_k, head_dim))
        elif self.optimization_strategy == 'memory':
            configs.extend(self._get_memory_optimized_configs(seq_len_q, seq_len_k, head_dim))
        
        # Add ML-predicted configurations if model is available
        if self.performance_model is not None:
            ml_configs = self._get_ml_predicted_configs(seq_len_q, seq_len_k, head_dim)
            configs.extend(ml_configs)
        
        # Multi-GPU specific configurations
        if num_gpus > 1:
            multi_gpu_configs = self._get_multi_gpu_configs(seq_len_q, seq_len_k, head_dim, num_gpus)
            configs.extend(multi_gpu_configs)
        
        # Remove duplicates and invalid configurations
        unique_configs = self._deduplicate_and_validate_configs(configs, seq_len_q, seq_len_k, head_dim)
        
        # Sort by predicted performance and return top candidates
        if self.performance_model is not None:
            unique_configs = self._sort_by_predicted_performance(unique_configs, seq_len_q, seq_len_k, head_dim)
        
        return unique_configs[:max_configs]
    
    def _get_small_sequence_configs(self, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations optimized for small sequences (≤512)"""
        configs = []
        
        block_options = [32, 48, 64]
        stages_options = [2, 3, 4]
        warps_options = [4, 6]
        
        for block_m, block_n in itertools.product(block_options, block_options):
            for stages, warps in itertools.product(stages_options, warps_options):
                # Small sequences benefit from smaller blocks and fewer stages
                if block_m * block_n > 2048:  # Avoid overly large blocks
                    continue
                
                config = EnhancedKernelConfig(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=head_dim,
                    NUM_STAGES=stages,
                    NUM_WARPS=warps,
                    enable_warp_specialization=True,
                    enable_double_buffering=stages >= 3,
                    enable_prefetch=True,
                    use_fast_math=False,
                )
                configs.append(config)
        
        return configs
    
    def _get_medium_sequence_configs(self, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations optimized for medium sequences (513-2048)"""
        configs = []
        
        # Prefer balanced block sizes with good Tensor Core utilization
        block_options = [48, 64, 80, 96]
        stages_options = [3, 4, 5]
        warps_options = [4, 6, 8]
        
        for block_m, block_n in itertools.product(block_options, block_options):
            for stages, warps in itertools.product(stages_options, warps_options):
                # Ensure blocks are Tensor Core friendly (multiples of 16)
                if block_m % 16 != 0 or block_n % 16 != 0:
                    continue
                
                config = EnhancedKernelConfig(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=head_dim,
                    NUM_STAGES=stages,
                    NUM_WARPS=warps,
                    enable_warp_specialization=True,
                    enable_double_buffering=True,
                    enable_prefetch=True,
                    use_fast_math=False,
                )
                configs.append(config)
        
        return configs
    
    def _get_large_sequence_configs(self, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations optimized for large sequences (2049-8192)"""
        configs = []
        
        # Prefer larger blocks with aggressive pipelining
        block_m_options = [64, 96, 128, 160]
        block_n_options = [64, 96, 128]
        stages_options = [4, 5, 6]
        warps_options = [6, 8]
        
        for block_m in block_m_options:
            for block_n in block_n_options:
                for stages, warps in itertools.product(stages_options, warps_options):
                    # Ensure memory fits in shared memory
                    estimated_memory = self._estimate_sram_usage(block_m, block_n, head_dim)
                    if estimated_memory > V100MemoryHierarchy.SRAM_SIZE * 0.9:
                        continue
                    
                    config = EnhancedKernelConfig(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=head_dim,
                        NUM_STAGES=stages,
                        NUM_WARPS=warps,
                        enable_warp_specialization=True,
                        enable_double_buffering=True,
                        enable_prefetch=True,
                        use_fast_math=False,
                    )
                    configs.append(config)
        
        return configs
    
    def _get_xlarge_sequence_configs(self, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations optimized for extra large sequences (>8192)"""
        configs = []
        
        # Focus on memory efficiency with large blocks
        block_m_options = [96, 128, 160, 192]
        block_n_options = [64, 96, 128]
        stages_options = [4, 5, 6]
        warps_options = [6, 8]
        
        for block_m in block_m_options:
            for block_n in block_n_options:
                for stages, warps in itertools.product(stages_options, warps_options):
                    # Memory constraint is critical for very large sequences
                    estimated_memory = self._estimate_sram_usage(block_m, block_n, head_dim)
                    if estimated_memory > V100MemoryHierarchy.SRAM_SIZE * 0.85:
                        continue
                    
                    config = EnhancedKernelConfig(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=head_dim,
                        NUM_STAGES=min(stages, 5),  # Limit stages for memory
                        NUM_WARPS=warps,
                        enable_warp_specialization=True,
                        enable_double_buffering=True,
                        enable_prefetch=True,
                        use_fast_math=self.optimization_strategy == 'speed',
                        memory_efficient=True,
                    )
                    configs.append(config)
        
        return configs
    
    def _get_speed_optimized_configs(self, seq_len_q: int, seq_len_k: int, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations specifically optimized for speed"""
        configs = []
        
        # Large blocks, high parallelism, aggressive optimizations
        base_sizes = [128, 160] if max(seq_len_q, seq_len_k) > 1024 else [64, 96]
        
        for block_m in base_sizes:
            for block_n in base_sizes:
                config = EnhancedKernelConfig(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=head_dim,
                    NUM_STAGES=6,  # Aggressive pipelining
                    NUM_WARPS=8,   # Maximum warps
                    enable_warp_specialization=True,
                    enable_double_buffering=True,
                    enable_prefetch=True,
                    use_fast_math=True,  # Speed over precision
                    memory_efficient=False,  # Speed over memory
                )
                
                # Check if configuration is feasible
                if self._estimate_sram_usage(block_m, block_n, head_dim) <= V100MemoryHierarchy.SRAM_SIZE:
                    configs.append(config)
        
        return configs
    
    def _get_memory_optimized_configs(self, seq_len_q: int, seq_len_k: int, head_dim: int) -> List[EnhancedKernelConfig]:
        """Get configurations specifically optimized for memory efficiency"""
        configs = []
        
        # Smaller blocks, conservative settings
        base_sizes = [32, 48, 64]
        
        for block_m in base_sizes:
            for block_n in base_sizes:
                config = EnhancedKernelConfig(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=head_dim,
                    NUM_STAGES=3,  # Conservative pipelining
                    NUM_WARPS=4,   # Fewer warps
                    enable_warp_specialization=False,  # Simpler execution
                    enable_double_buffering=False,     # Save memory
                    enable_prefetch=False,             # Save memory
                    use_fast_math=False,
                    memory_efficient=True,
                )
                configs.append(config)
        
        return configs
    
    def _get_ml_predicted_configs(self, seq_len_q: int, seq_len_k: int, head_dim: int) -> List[EnhancedKernelConfig]:
        """Generate configurations based on ML model predictions"""
        if self.performance_model is None:
            return []
        
        # This would use a trained ML model to predict good configurations
        # For now, return empty list (placeholder for future ML implementation)
        return []
    
    def _get_multi_gpu_configs(self, seq_len_q: int, seq_len_k: int, head_dim: int, num_gpus: int) -> List[EnhancedKernelConfig]:
        """Get configurations optimized for multi-GPU setups"""
        configs = []
        
        if num_gpus <= 1:
            return configs
        
        # Multi-GPU typically benefits from larger blocks to reduce communication
        block_multiplier = min(2, num_gpus // 2)
        base_block = 64
        
        for multiplier in [1, block_multiplier]:
            block_m = base_block * multiplier
            block_n = base_block * multiplier
            
            if block_m > 160 or block_n > 160:  # Practical limits
                continue
            
            config = EnhancedKernelConfig(
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=head_dim,
                NUM_STAGES=4,
                NUM_WARPS=6,
                enable_warp_specialization=True,
                enable_double_buffering=True,
                enable_prefetch=True,
                use_fast_math=False,
            )
            
            if self._estimate_sram_usage(block_m, block_n, head_dim) <= V100MemoryHierarchy.SRAM_SIZE:
                configs.append(config)
        
        return configs
    
    def _deduplicate_and_validate_configs(self, 
                                        configs: List[EnhancedKernelConfig], 
                                        seq_len_q: int, 
                                        seq_len_k: int, 
                                        head_dim: int) -> List[EnhancedKernelConfig]:
        """Remove duplicate and invalid configurations"""
        seen_keys = set()
        valid_configs = []
        
        for config in configs:
            key = config.get_cache_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            # Validate configuration
            if self._validate_config(config, seq_len_q, seq_len_k, head_dim):
                valid_configs.append(config)
        
        return valid_configs
    
    def _validate_config(self, config: EnhancedKernelConfig, seq_len_q: int, seq_len_k: int, head_dim: int) -> bool:
        """Validate if configuration is feasible"""
        # Check basic constraints
        if config.BLOCK_M <= 0 or config.BLOCK_N <= 0 or config.BLOCK_K != head_dim:
            return False
        
        if config.BLOCK_M > seq_len_q or config.BLOCK_N > seq_len_k:
            return False
        
        if config.NUM_WARPS not in [1, 2, 4, 6, 8, 12, 16]:
            return False
        
        if config.NUM_STAGES < 1 or config.NUM_STAGES > 8:
            return False
        
        # Check memory constraints
        estimated_memory = self._estimate_sram_usage(config.BLOCK_M, config.BLOCK_N, head_dim)
        if estimated_memory > V100MemoryHierarchy.SRAM_SIZE:
            return False
        
        # Check Tensor Core alignment for optimal performance
        if config.BLOCK_M % 16 != 0 or config.BLOCK_N % 16 != 0:
            return False
        
        return True
    
    def _sort_by_predicted_performance(self, 
                                     configs: List[EnhancedKernelConfig], 
                                     seq_len_q: int, 
                                     seq_len_k: int, 
                                     head_dim: int) -> List[EnhancedKernelConfig]:
        """Sort configurations by predicted performance"""
        # Simple heuristic-based scoring (replace with ML model predictions later)
        def score_config(config):
            # Prefer Tensor Core friendly sizes
            tc_score = 1.0 if (config.BLOCK_M % 16 == 0 and config.BLOCK_N % 16 == 0) else 0.8
            
            # Prefer balanced block sizes
            block_ratio = min(config.BLOCK_M, config.BLOCK_N) / max(config.BLOCK_M, config.BLOCK_N)
            balance_score = block_ratio
            
            # Memory utilization score
            memory_usage = self._estimate_sram_usage(config.BLOCK_M, config.BLOCK_N, head_dim)
            memory_utilization = memory_usage / V100MemoryHierarchy.SRAM_SIZE
            memory_score = 1.0 - abs(memory_utilization - 0.7)  # Target ~70% utilization
            
            # Pipeline efficiency
            pipeline_score = min(config.NUM_STAGES / 6.0, 1.0)
            
            # Warp efficiency
            warp_score = min(config.NUM_WARPS / 8.0, 1.0)
            
            total_score = (tc_score * 0.3 + balance_score * 0.2 + 
                          memory_score * 0.3 + pipeline_score * 0.1 + 
                          warp_score * 0.1)
            
            return total_score
        
        configs_with_scores = [(config, score_config(config)) for config in configs]
        configs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [config for config, _ in configs_with_scores]
    
    def _estimate_sram_usage(self, block_m: int, block_n: int, head_dim: int) -> int:
        """Estimate shared memory usage for given block configuration"""
        element_size = 2  # Assume fp16
        
        # Q block: [BLOCK_M, head_dim]
        q_block_size = block_m * head_dim * element_size
        
        # K block: [BLOCK_N, head_dim] 
        k_block_size = block_n * head_dim * element_size
        
        # V block: [BLOCK_N, head_dim]
        v_block_size = block_n * head_dim * element_size
        
        # Attention scores: [BLOCK_M, BLOCK_N] (fp32 for numerical stability)
        attention_scores_size = block_m * block_n * 4
        
        # Output accumulator: [BLOCK_M, head_dim] (fp32)
        output_acc_size = block_m * head_dim * 4
        
        # Statistics arrays: [BLOCK_M] each (fp32)
        statistics_size = 2 * block_m * 4
        
        # Double buffering overhead if enabled
        double_buffer_overhead = (k_block_size + v_block_size)
        
        # Pipeline overhead
        pipeline_overhead = attention_scores_size * 0.2  # 20% overhead for pipelining
        
        total_size = (q_block_size + k_block_size + v_block_size + 
                     attention_scores_size + output_acc_size + statistics_size +
                     double_buffer_overhead + pipeline_overhead)
        
        return int(total_size * 1.1)  # 10% safety margin
    
    def benchmark_config_enhanced(
        self, 
        config: EnhancedKernelConfig,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float,
        causal: bool,
        num_warmup: int = 10,
        num_runs: int = 50,
        measure_accuracy: bool = True,
    ) -> EnhancedBenchmarkResult:
        """
        Enhanced benchmark of a specific kernel configuration
        
        Args:
            config: Kernel configuration to benchmark
            q, k, v: Input tensors
            scale: Attention scale
            causal: Whether to use causal masking
            num_warmup: Number of warmup runs
            num_runs: Number of timed runs
            measure_accuracy: Whether to measure numerical accuracy
            
        Returns:
            Enhanced benchmark result with detailed metrics
        """
        from .forward_kernel import flash_attention_forward_triton_enhanced
        
        try:
            # Warmup runs
            for _ in range(num_warmup):
                _ = flash_attention_forward_triton_enhanced(
                    q, k, v, scale, causal, 0.0, None,
                    config.BLOCK_M, config.BLOCK_N,
                    config.enable_warp_specialization,
                    config.enable_double_buffering,
                    config.enable_prefetch,
                    config.use_fast_math
                )
            
            torch.cuda.synchronize()
            
            # Measure peak memory
            torch.cuda.reset_peak_memory_stats()
            
            # Timed runs with detailed measurement
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                output, lse, max_vals = flash_attention_forward_triton_enhanced(
                    q, k, v, scale, causal, 0.0, None,
                    config.BLOCK_M, config.BLOCK_N,
                    config.enable_warp_specialization,
                    config.enable_double_buffering,
                    config.enable_prefetch,
                    config.use_fast_math
                )
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time_ms = np.mean(times)
            std_time_ms = np.std(times)
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            
            # Calculate performance metrics
            batch_size, seq_len_q, num_heads, head_dim = q.shape
            _, seq_len_k, _, _ = k.shape
            
            # FLOPS calculation (approximate for attention)
            flops = 4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
            throughput_tflops = (flops / (avg_time_ms / 1000)) / 1e12
            
            # Memory bandwidth calculation
            total_memory_bytes = (q.numel() + k.numel() + v.numel() + output.numel()) * q.element_size()
            memory_bandwidth_gbps = (total_memory_bytes / (avg_time_ms / 1000)) / 1e9
            
            # Estimate Tensor Core utilization (heuristic)
            theoretical_tc_tflops = PerformanceConfig.TARGET_METRICS['peak_tflops']
            tensor_core_utilization = min(throughput_tflops / theoretical_tc_tflops, 1.0)
            
            # Estimate occupancy (simplified)
            estimated_memory = self._estimate_sram_usage(config.BLOCK_M, config.BLOCK_N, head_dim)
            occupancy_percentage = min(100, 
                (V100MemoryHierarchy.SRAM_SIZE / estimated_memory) * 
                (config.NUM_WARPS / V100MemoryHierarchy.MAX_WARPS_PER_SM) * 100)
            
            # Numerical accuracy measurement
            numerical_accuracy = 1.0
            if measure_accuracy:
                try:
                    # Compare with PyTorch reference (simplified)
                    numerical_accuracy = self._measure_numerical_accuracy(q, k, v, output, scale, causal)
                except:
                    numerical_accuracy = 0.9  # Default if measurement fails
            
            return EnhancedBenchmarkResult(
                config=config,
                avg_time_ms=avg_time_ms,
                std_time_ms=std_time_ms,
                throughput_tflops=throughput_tflops,
                memory_bandwidth_gbps=memory_bandwidth_gbps,
                tensor_core_utilization=tensor_core_utilization,
                occupancy_percentage=occupancy_percentage,
                success=True,
                peak_memory_mb=peak_memory_mb,
                numerical_accuracy=numerical_accuracy,
            )
            
        except Exception as e:
            return EnhancedBenchmarkResult(
                config=config,
                avg_time_ms=float('inf'),
                std_time_ms=0.0,
                throughput_tflops=0.0,
                memory_bandwidth_gbps=0.0,
                tensor_core_utilization=0.0,
                occupancy_percentage=0.0,
                success=False,
                error_msg=str(e),
            )
    
    def _measure_numerical_accuracy(self, q, k, v, output, scale, causal) -> float:
        """Measure numerical accuracy against reference implementation"""
        try:
            import torch.nn.functional as F
            
            # Simplified accuracy measurement
            with torch.no_grad():
                q_ref = q.transpose(1, 2).contiguous()
                k_ref = k.transpose(1, 2).contiguous()
                v_ref = v.transpose(1, 2).contiguous()
                
                ref_output = F.scaled_dot_product_attention(
                    q_ref, k_ref, v_ref, is_causal=causal, scale=scale
                )
                ref_output = ref_output.transpose(1, 2).contiguous()
                
                # Calculate relative error
                max_error = torch.max(torch.abs(output - ref_output))
                relative_error = max_error / (torch.max(torch.abs(ref_output)) + 1e-8)
                
                # Convert to accuracy score
                accuracy = max(0.0, 1.0 - relative_error.item())
                return accuracy
        except:
            return 0.9  # Default accuracy if measurement fails
    
    def tune_enhanced(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        causal: bool = False,
        num_gpus: int = 1,
        use_cache: bool = True,
        max_configs: int = 20,
        parallel_benchmark: bool = True,
    ) -> EnhancedKernelConfig:
        """
        Enhanced tuning with intelligent configuration selection and parallel benchmarking
        
        Args:
            q, k, v: Input tensors
            scale: Attention scale
            causal: Whether to use causal masking
            num_gpus: Number of GPUs available
            use_cache: Whether to use cached results
            max_configs: Maximum configurations to benchmark
            parallel_benchmark: Whether to benchmark configurations in parallel
            
        Returns:
            Optimal kernel configuration
        """
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        cache_key = self._get_cache_key(
            batch_size, seq_len_q, seq_len_k, num_heads, head_dim, 
            str(q.dtype), causal, num_gpus
        )
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            cached_config_dict = cached_entry['config']
            
            cached_config = EnhancedKernelConfig(**cached_config_dict)
            print(f"Using cached configuration: {cached_config}")
            return cached_config
        
        # Generate intelligent candidate configurations
        candidates = self.get_intelligent_candidate_configs(
            seq_len_q, seq_len_k, head_dim, num_gpus, max_configs
        )
        
        if not candidates:
            # Fallback to default configuration
            from ..config import get_enhanced_config
            default_config_dict = get_enhanced_config(seq_len_q, head_dim, num_gpus, self.optimization_strategy)
            return EnhancedKernelConfig(**default_config_dict)
        
        print(f"Enhanced autotuning Flash Attention for shape {q.shape} with {len(candidates)} configurations...")
        print(f"Optimization strategy: {self.optimization_strategy}")
        
        # Benchmark configurations
        if parallel_benchmark and len(candidates) > 4:
            results = self._benchmark_parallel(candidates, q, k, v, scale, causal)
        else:
            results = self._benchmark_sequential(candidates, q, k, v, scale, causal)
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.numerical_accuracy > 0.95]
        
        if not successful_results:
            print("Warning: No configurations succeeded with good accuracy, using best available")
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                print("Warning: No configurations succeeded, using default")
                from ..config import get_enhanced_config
                default_config_dict = get_enhanced_config(seq_len_q, head_dim, num_gpus, self.optimization_strategy)
                return EnhancedKernelConfig(**default_config_dict)
        
        # Select best configuration based on strategy
        best_result = self._select_best_result(successful_results)
        best_config = best_result.config
        
        # Print results
        print(f"Best configuration: {best_config}")
        print(f"  Performance score: {best_result.get_performance_score():.3f}")
        print(f"  Time: {best_result.avg_time_ms:.2f} ± {best_result.std_time_ms:.2f} ms")
        print(f"  Throughput: {best_result.throughput_tflops:.2f} TFLOPS")
        print(f"  Memory BW: {best_result.memory_bandwidth_gbps:.1f} GB/s")
        print(f"  TC Utilization: {best_result.tensor_core_utilization:.1%}")
        print(f"  Occupancy: {best_result.occupancy_percentage:.1f}%")
        print(f"  Numerical accuracy: {best_result.numerical_accuracy:.3f}")
        
        # Cache the result
        if use_cache:
            self.cache[cache_key] = {
                'config': best_config.__dict__,
                'performance': {
                    'avg_time_ms': best_result.avg_time_ms,
                    'throughput_tflops': best_result.throughput_tflops,
                    'performance_score': best_result.get_performance_score(),
                },
                'timestamp': time.time(),
                'device_info': {
                    'capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                }
            }
            self._save_cache()
        
        # Update performance model with new data
        self._update_performance_model(successful_results)
        
        return best_config
    
    def _benchmark_sequential(self, candidates, q, k, v, scale, causal) -> List[EnhancedBenchmarkResult]:
        """Benchmark configurations sequentially"""
        results = []
        for i, config in enumerate(candidates):
            if i % 5 == 0:
                print(f"  Progress: {i+1}/{len(candidates)}")
            
            result = self.benchmark_config_enhanced(config, q, k, v, scale, causal)
            results.append(result)
        
        return results
    
    def _benchmark_parallel(self, candidates, q, k, v, scale, causal) -> List[EnhancedBenchmarkResult]:
        """Benchmark configurations in parallel"""
        results = []
        
        # Note: True parallel benchmarking on GPU is complex due to resource sharing
        # This is a simplified version that could be enhanced with proper GPU resource management
        with ThreadPoolExecutor(max_workers=2) as executor:  # Limited parallelism for GPU
            future_to_config = {
                executor.submit(self.benchmark_config_enhanced, config, q, k, v, scale, causal): config
                for config in candidates
            }
            
            for i, future in enumerate(as_completed(future_to_config)):
                if i % 5 == 0:
                    print(f"  Progress: {i+1}/{len(candidates)}")
                
                result = future.result()
                results.append(result)
        
        return results
    
    def _select_best_result(self, results: List[EnhancedBenchmarkResult]) -> EnhancedBenchmarkResult:
        """Select best result based on optimization strategy"""
        if self.optimization_strategy == 'speed':
            weights = {'weight_speed': 0.7, 'weight_memory': 0.1, 'weight_efficiency': 0.2}
        elif self.optimization_strategy == 'memory':
            weights = {'weight_speed': 0.1, 'weight_memory': 0.7, 'weight_efficiency': 0.2}
        elif self.optimization_strategy == 'balanced':
            weights = {'weight_speed': 0.4, 'weight_memory': 0.3, 'weight_efficiency': 0.3}
        else:  # adaptive
            # Adaptive strategy based on current results
            avg_memory = np.mean([r.peak_memory_mb for r in results])
            if avg_memory > 12000:  # High memory usage
                weights = {'weight_speed': 0.2, 'weight_memory': 0.6, 'weight_efficiency': 0.2}
            else:
                weights = {'weight_speed': 0.5, 'weight_memory': 0.2, 'weight_efficiency': 0.3}
        
        # Calculate scores and select best
        best_result = max(results, key=lambda r: r.get_performance_score(**weights))
        return best_result
    
    def _update_performance_model(self, results: List[EnhancedBenchmarkResult]):
        """Update ML performance model with new benchmark data"""
        if not self.enable_predictive_modeling:
            return
        
        # Add results to history
        self.benchmark_history.extend(results)
        
        # Simple placeholder for ML model update
        # In a full implementation, this would train a regression model
        # to predict performance based on configuration parameters
        pass


# Legacy compatibility and convenience functions
def get_autotuning_config(
    seq_len_q: int, 
    seq_len_k: int, 
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """
    Get pre-tuned configuration for common workloads (enhanced version)
    
    This function provides reasonable configurations without running
    full autotuning, based on empirical results and heuristics.
    """
    from ..config import get_enhanced_config
    
    # Use enhanced configuration system
    enhanced_config = get_enhanced_config(
        max(seq_len_q, seq_len_k), head_dim, 1, 'balanced'
    )
    
    return enhanced_config


def benchmark_kernel_configs(
    configs: List[EnhancedKernelConfig],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False
) -> List[EnhancedBenchmarkResult]:
    """
    Benchmark multiple kernel configurations with enhanced metrics
    """
    tuner = AdvancedAutoTuner()
    results = []
    
    for config in configs:
        result = tuner.benchmark_config_enhanced(config, q, k, v, scale, causal)
        results.append(result)
    
    return results


# Export main classes and functions
__all__ = [
    'EnhancedKernelConfig',
    'EnhancedBenchmarkResult',
    'AdvancedAutoTuner',
    'get_autotuning_config',
    'benchmark_kernel_configs',
]