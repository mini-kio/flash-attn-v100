# interface.py
"""
Enhanced user interface for Flash Attention V100

This module provides the main API with advanced features:
- Multi-GPU support and automatic device management
- Enhanced error handling and recovery
- Performance monitoring and adaptive optimization
- Comprehensive validation and debugging
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import time
import contextlib
import os

from .utils import validate_inputs, calculate_memory_requirements
from .config import get_enhanced_config, DebugConfig, OptimizationConfig, MultiGPUConfig
from .ops.attention import FlashAttentionV100Function


class FlashAttentionV100Manager:
    """
    Advanced manager for Flash Attention V100 with intelligent optimization
    
    Features:
    - Automatic device management and multi-GPU coordination
    - Performance monitoring and adaptive configuration
    - Memory usage optimization
    - Error recovery and fallback mechanisms
    """
    
    def __init__(self, 
                 enable_multi_gpu: bool = None,
                 optimization_strategy: str = 'balanced',
                 enable_profiling: bool = False):
        """
        Initialize Flash Attention V100 Manager
        
        Args:
            enable_multi_gpu: Whether to enable multi-GPU support (auto-detect if None)
            optimization_strategy: 'speed', 'memory', 'balanced', or 'adaptive'
            enable_profiling: Whether to enable performance profiling
        """
        self.optimization_strategy = optimization_strategy
        self.enable_profiling = enable_profiling
        
        # Auto-detect multi-GPU capability
        if enable_multi_gpu is None:
            self.enable_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        else:
            self.enable_multi_gpu = enable_multi_gpu and torch.cuda.is_available()
        
        # Device management
        self.devices = self._get_available_devices()
        self.primary_device = self.devices[0] if self.devices else torch.device('cpu')
        
        # Performance tracking
        self.performance_history = []
        self.config_cache = {}
        self.error_recovery_count = 0
        
        # Multi-GPU setup
        if self.enable_multi_gpu and len(self.devices) > 1:
            self._setup_multi_gpu()
        
        print(f"Flash Attention V100 Manager initialized:")
        print(f"  Devices: {len(self.devices)} ({'Multi-GPU' if self.enable_multi_gpu else 'Single-GPU'})")
        print(f"  Strategy: {self.optimization_strategy}")
        print(f"  Profiling: {'Enabled' if self.enable_profiling else 'Disabled'}")
    
    def _get_available_devices(self) -> List[torch.device]:
        """Get list of available CUDA devices with V100+ capability"""
        devices = []
        
        if not torch.cuda.is_available():
            return [torch.device('cpu')]
        
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            if capability >= (7, 0):  # V100 or newer
                devices.append(torch.device(f'cuda:{i}'))
        
        if not devices:
            # Fallback to any available CUDA device
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        
        return devices
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        if not dist.is_initialized():
            try:
                # Initialize process group if not already done
                if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                    dist.init_process_group(backend='nccl')
                    print("Multi-GPU: Using existing distributed environment")
                else:
                    # Single-node multi-GPU setup
                    print("Multi-GPU: Setting up single-node multi-GPU")
            except Exception as e:
                print(f"Warning: Multi-GPU setup failed: {e}")
                self.enable_multi_gpu = False
    
    def get_optimal_config(self, 
                          seq_len_q: int, 
                          seq_len_k: int, 
                          num_heads: int, 
                          head_dim: int,
                          force_recompute: bool = False) -> Dict[str, Any]:
        """
        Get optimal configuration for given parameters
        
        Args:
            seq_len_q, seq_len_k: Sequence lengths
            num_heads: Number of attention heads
            head_dim: Head dimension
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Optimal configuration dictionary
        """
        cache_key = f"{seq_len_q}_{seq_len_k}_{num_heads}_{head_dim}_{len(self.devices)}"
        
        if not force_recompute and cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        # Get base configuration
        config = get_enhanced_config(
            max(seq_len_q, seq_len_k), 
            head_dim, 
            len(self.devices),
            self.optimization_strategy
        )
        
        # Adaptive adjustments based on performance history
        if len(self.performance_history) > 5:
            config = self._adapt_config_from_history(config, seq_len_q, seq_len_k, head_dim)
        
        # Multi-GPU specific adjustments
        if self.enable_multi_gpu and len(self.devices) > 1:
            config = self._adjust_config_for_multi_gpu(config, num_heads)
        
        self.config_cache[cache_key] = config
        return config
    
    def _adapt_config_from_history(self, config, seq_len_q, seq_len_k, head_dim):
        """Adapt configuration based on performance history"""
        # Analyze recent performance trends
        recent_history = self.performance_history[-10:]
        
        avg_memory_usage = sum(h.get('peak_memory_mb', 0) for h in recent_history) / len(recent_history)
        avg_throughput = sum(h.get('throughput_tflops', 0) for h in recent_history) / len(recent_history)
        
        # Adjust strategy if needed
        if avg_memory_usage > 14000:  # High memory usage (>14GB)
            config['BLOCK_M'] = min(config['BLOCK_M'], 64)
            config['BLOCK_N'] = min(config['BLOCK_N'], 64)
            config['NUM_STAGES'] = min(config['NUM_STAGES'], 3)
        elif avg_throughput < 30:  # Low throughput
            config['BLOCK_M'] = min(config['BLOCK_M'] * 2, 128)
            config['BLOCK_N'] = min(config['BLOCK_N'] * 2, 128)
            config['NUM_STAGES'] = min(config['NUM_STAGES'] + 1, 6)
        
        return config
    
    def _adjust_config_for_multi_gpu(self, config, num_heads):
        """Adjust configuration for multi-GPU setup"""
        num_gpus = len(self.devices)
        
        # Head parallelization strategy
        if num_heads % num_gpus == 0:
            config['PARALLELIZATION_STRATEGY'] = 'head_parallel'
            config['HEADS_PER_GPU'] = num_heads // num_gpus
        else:
            config['PARALLELIZATION_STRATEGY'] = 'sequence_parallel'
        
        # Adjust block sizes for better communication efficiency
        config['BLOCK_M'] = max(config['BLOCK_M'], 64)  # Larger blocks for multi-GPU
        config['BLOCK_N'] = max(config['BLOCK_N'], 64)
        
        return config


# Global manager instance
_global_manager = None

def get_flash_attention_manager(**kwargs) -> FlashAttentionV100Manager:
    """Get global Flash Attention manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = FlashAttentionV100Manager(**kwargs)
    return _global_manager


def flash_attention_v100(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    block_size_m: Optional[int] = None,
    block_size_n: Optional[int] = None,
    enable_autotuning: bool = True,
    return_softmax_lse: bool = False,
    # Enhanced parameters
    optimization_strategy: str = 'balanced',
    enable_multi_gpu: Optional[bool] = None,
    enable_profiling: bool = False,
    fallback_on_error: bool = True,
    validate_inputs: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Enhanced Flash Attention implementation optimized for V100 GPUs
    
    This function implements the Flash Attention algorithm with advanced optimizations:
    - Multi-GPU support with automatic parallelization
    - Adaptive configuration based on performance history
    - Enhanced error handling with automatic fallback
    - Comprehensive input validation and memory optimization
    
    Args:
        q: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]  
        v: Value tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
        causal: If True, apply causal masking (lower triangular)
        scale: Attention scale factor. If None, defaults to 1/sqrt(head_dim)
        dropout_p: Dropout probability (0.0 to disable)
        bias: Optional bias tensor for attention scores
        block_size_m: Block size for M dimension (Q sequence length)
        block_size_n: Block size for N dimension (K/V sequence length)
        enable_autotuning: Whether to use intelligent autotuning
        return_softmax_lse: Whether to return log-sum-exp values for debugging
        
        # Enhanced parameters
        optimization_strategy: 'speed', 'memory', 'balanced', or 'adaptive'
        enable_multi_gpu: Whether to enable multi-GPU (auto-detect if None)
        enable_profiling: Whether to enable performance profiling
        fallback_on_error: Whether to fallback to PyTorch on errors
        validate_inputs: Whether to perform comprehensive input validation
        
    Returns:
        Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        If return_softmax_lse=True, also returns LSE tensor
        
    Example:
        >>> import torch
        >>> from flash_attention_v100 import flash_attention_v100
        >>> 
        >>> # Basic usage
        >>> batch_size, seq_len, num_heads, head_dim = 2, 1024, 12, 64
        >>> q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        >>> 
        >>> # Enhanced usage with multi-GPU and profiling
        >>> output = flash_attention_v100(
        ...     q, k, v, causal=True, 
        ...     optimization_strategy='speed',
        ...     enable_multi_gpu=True,
        ...     enable_profiling=True
        ... )
        >>> 
        >>> # With dropout and bias
        >>> bias = torch.randn(1, num_heads, seq_len, seq_len, device='cuda', dtype=torch.float16)
        >>> output = flash_attention_v100(q, k, v, dropout_p=0.1, bias=bias)
    """
    
    # Get or create manager
    manager = get_flash_attention_manager(
        enable_multi_gpu=enable_multi_gpu,
        optimization_strategy=optimization_strategy,
        enable_profiling=enable_profiling
    )
    
    # Performance monitoring context
    perf_start_time = time.time()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        # Input validation
        if validate_inputs:
            q, k, v, scale = validate_inputs_enhanced(q, k, v, scale, bias, dropout_p)
        
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, num_heads_kv, _ = k.shape
        
        # Get optimal configuration
        if enable_autotuning:
            config = manager.get_optimal_config(seq_len_q, seq_len_k, num_heads, head_dim)
            if block_size_m is None:
                block_size_m = config['BLOCK_M']
            if block_size_n is None:
                block_size_n = config['BLOCK_N']
        
        # Memory usage prediction and warnings
        if DebugConfig.LOG_MEMORY_USAGE:
            memory_info = calculate_memory_requirements(
                batch_size, max(seq_len_q, seq_len_k), num_heads, head_dim, q.dtype
            )
            print(f"Predicted memory usage: {memory_info['total_required_mb']:.1f} MB")
        
        # Check for potential issues
        _check_potential_issues(q, k, v, seq_len_q, seq_len_k, causal)
        
        # Multi-GPU execution
        if manager.enable_multi_gpu and len(manager.devices) > 1:
            result = _execute_multi_gpu(
                q, k, v, scale, causal, dropout_p, bias,
                block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
                manager, config
            )
        else:
            # Single GPU execution
            result = FlashAttentionV100Function.apply(
                q, k, v, scale, causal, dropout_p, bias,
                block_size_m, block_size_n, enable_autotuning, return_softmax_lse
            )
        
        # Performance tracking
        if manager.enable_profiling:
            _record_performance(manager, perf_start_time, initial_memory, q.shape, True)
        
        return result
        
    except Exception as e:
        # Error handling and recovery
        if fallback_on_error:
            warning_msg = f"Flash Attention failed ({str(e)}), falling back to PyTorch implementation"
            warnings.warn(warning_msg, UserWarning)
            
            # Fallback to PyTorch scaled_dot_product_attention
            result = _fallback_to_pytorch(q, k, v, scale, causal, dropout_p, bias, return_softmax_lse)
            
            # Record failed attempt
            if manager.enable_profiling:
                _record_performance(manager, perf_start_time, initial_memory, q.shape, False)
            
            manager.error_recovery_count += 1
            return result
        else:
            # Re-raise the exception if fallback is disabled
            raise e


def _execute_multi_gpu(q, k, v, scale, causal, dropout_p, bias,
                      block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
                      manager, config):
    """Execute Flash Attention across multiple GPUs"""
    
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    num_gpus = len(manager.devices)
    
    # Determine parallelization strategy
    strategy = config.get('PARALLELIZATION_STRATEGY', 'head_parallel')
    
    if strategy == 'head_parallel' and num_heads % num_gpus == 0:
        return _execute_head_parallel(
            q, k, v, scale, causal, dropout_p, bias,
            block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
            manager, config
        )
    elif strategy == 'sequence_parallel':
        return _execute_sequence_parallel(
            q, k, v, scale, causal, dropout_p, bias,
            block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
            manager, config
        )
    else:
        # Fallback to single GPU if parallelization not feasible
        return FlashAttentionV100Function.apply(
            q, k, v, scale, causal, dropout_p, bias,
            block_size_m, block_size_n, enable_autotuning, return_softmax_lse
        )


def _execute_head_parallel(q, k, v, scale, causal, dropout_p, bias,
                          block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
                          manager, config):
    """Execute with head parallelization across GPUs"""
    
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    num_gpus = len(manager.devices)
    heads_per_gpu = num_heads // num_gpus
    
    # Split tensors across heads
    q_splits = torch.chunk(q, num_gpus, dim=2)
    k_splits = torch.chunk(k, num_gpus, dim=2)  
    v_splits = torch.chunk(v, num_gpus, dim=2)
    
    if bias is not None:
        bias_splits = torch.chunk(bias, num_gpus, dim=1)
    else:
        bias_splits = [None] * num_gpus
    
    # Execute on each GPU
    outputs = []
    lses = [] if return_softmax_lse else None
    
    for i, device in enumerate(manager.devices):
        with torch.cuda.device(device):
            # Move tensors to current device
            q_local = q_splits[i].to(device, non_blocking=True)
            k_local = k_splits[i].to(device, non_blocking=True)
            v_local = v_splits[i].to(device, non_blocking=True)
            bias_local = bias_splits[i].to(device, non_blocking=True) if bias_splits[i] is not None else None
            
            # Execute Flash Attention
            result = FlashAttentionV100Function.apply(
                q_local, k_local, v_local, scale, causal, dropout_p, bias_local,
                block_size_m, block_size_n, enable_autotuning, return_softmax_lse
            )
            
            if return_softmax_lse:
                output_local, lse_local = result
                outputs.append(output_local.to(manager.primary_device, non_blocking=True))
                lses.append(lse_local.to(manager.primary_device, non_blocking=True))
            else:
                outputs.append(result.to(manager.primary_device, non_blocking=True))
    
    # Synchronize and concatenate results
    torch.cuda.synchronize()
    final_output = torch.cat(outputs, dim=2)
    
    if return_softmax_lse:
        final_lse = torch.cat(lses, dim=1)
        return final_output, final_lse
    else:
        return final_output


def _execute_sequence_parallel(q, k, v, scale, causal, dropout_p, bias,
                             block_size_m, block_size_n, enable_autotuning, return_softmax_lse,
                             manager, config):
    """Execute with sequence parallelization across GPUs"""
    
    # Sequence parallelization is more complex and requires careful handling of attention computation
    # For now, fallback to single GPU
    # TODO: Implement proper sequence parallelization with ring attention or similar
    
    warnings.warn("Sequence parallelization not yet implemented, using single GPU", UserWarning)
    return FlashAttentionV100Function.apply(
        q, k, v, scale, causal, dropout_p, bias,
        block_size_m, block_size_n, enable_autotuning, return_softmax_lse
    )


def _fallback_to_pytorch(q, k, v, scale, causal, dropout_p, bias, return_softmax_lse):
    """Fallback to PyTorch's scaled_dot_product_attention"""
    import torch.nn.functional as F
    
    # Convert to PyTorch's expected format: [batch, heads, seq, head_dim]
    q_pt = q.transpose(1, 2)
    k_pt = k.transpose(1, 2)  
    v_pt = v.transpose(1, 2)
    
    # Apply bias if provided
    if bias is not None:
        # PyTorch expects bias in [batch, heads, seq_q, seq_k] format
        if bias.dim() == 4:
            bias_pt = bias.squeeze(0) if bias.size(0) == 1 else bias
        else:
            bias_pt = bias
    else:
        bias_pt = None
    
    # Use PyTorch's implementation
    output_pt = F.scaled_dot_product_attention(
        q_pt, k_pt, v_pt,
        attn_mask=bias_pt,
        dropout_p=dropout_p if q.training else 0.0,
        is_causal=causal,
        scale=scale
    )
    
    # Convert back to our format: [batch, seq, heads, head_dim]
    output = output_pt.transpose(1, 2)
    
    if return_softmax_lse:
        # Create dummy LSE tensor since PyTorch doesn't return it
        batch_size, seq_len, num_heads, _ = output.shape
        dummy_lse = torch.zeros(batch_size, num_heads, seq_len, device=output.device, dtype=torch.float32)
        return output, dummy_lse
    else:
        return output


def validate_inputs_enhanced(q, k, v, scale, bias, dropout_p):
    """Enhanced input validation with comprehensive checks"""
    from .utils import validate_inputs as base_validate
    
    # Basic validation
    q, k, v, scale = base_validate(q, k, v, scale)
    
    # Additional validations
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
    
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a torch.Tensor")
        
        batch_size, seq_len_q, num_heads, _ = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Check bias shape compatibility
        expected_shapes = [
            (batch_size, num_heads, seq_len_q, seq_len_k),  # Full bias
            (1, num_heads, seq_len_q, seq_len_k),           # Broadcast batch
            (batch_size, 1, seq_len_q, seq_len_k),          # Broadcast heads
            (1, 1, seq_len_q, seq_len_k),                   # Broadcast both
        ]
        
        if bias.shape not in expected_shapes:
            raise ValueError(f"bias shape {bias.shape} not compatible with attention shape. "
                           f"Expected one of: {expected_shapes}")
        
        if bias.device != q.device:
            raise ValueError("bias must be on the same device as input tensors")
        
        if bias.dtype != q.dtype:
            warnings.warn(f"bias dtype ({bias.dtype}) differs from input dtype ({q.dtype}). "
                         "This may cause precision issues.")
    
    return q, k, v, scale


def _check_potential_issues(q, k, v, seq_len_q, seq_len_k, causal):
    """Check for potential performance or correctness issues"""
    
    # Check for very large sequences that might cause OOM
    if seq_len_q > 16384 or seq_len_k > 16384:
        warnings.warn(
            f"Very large sequence lengths detected (Q: {seq_len_q}, K: {seq_len_k}). "
            "Consider using gradient checkpointing or sequence parallelization.",
            UserWarning
        )
    
    # Check for inefficient shapes
    batch_size, _, num_heads, head_dim = q.shape
    
    if num_heads % 4 != 0:
        warnings.warn(
            f"Number of heads ({num_heads}) is not divisible by 4. "
            "This may lead to suboptimal Tensor Core utilization.",
            UserWarning
        )
    
    if head_dim % 16 != 0:
        warnings.warn(
            f"Head dimension ({head_dim}) is not divisible by 16. "
            "This may lead to suboptimal Tensor Core utilization.",
            UserWarning
        )
    
    # Check memory requirements
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory
        estimated_usage = calculate_memory_requirements(
            batch_size, max(seq_len_q, seq_len_k), num_heads, head_dim, q.dtype
        )
        
        if estimated_usage['total_required_mb'] * 1024**2 > available_memory * 0.9:
            warnings.warn(
                f"Estimated memory usage ({estimated_usage['total_required_mb']:.1f} MB) "
                f"is close to GPU capacity. Consider reducing batch size or sequence length.",
                UserWarning
            )


def _record_performance(manager, start_time, initial_memory, input_shape, success):
    """Record performance metrics for adaptive optimization"""
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        current_memory_mb = torch.cuda.memory_allocated() / (1024**2)
    else:
        peak_memory_mb = current_memory_mb = 0
    
    # Calculate approximate throughput
    batch_size, seq_len_q, num_heads, head_dim = input_shape
    total_elements = batch_size * seq_len_q * num_heads * head_dim
    throughput_tflops = (4 * total_elements / (execution_time_ms / 1000)) / 1e12 if success else 0
    
    performance_record = {
        'timestamp': end_time,
        'input_shape': input_shape,
        'execution_time_ms': execution_time_ms,
        'peak_memory_mb': peak_memory_mb,
        'current_memory_mb': current_memory_mb,
        'throughput_tflops': throughput_tflops,
        'success': success,
        'optimization_strategy': manager.optimization_strategy,
        'num_devices': len(manager.devices),
    }
    
    manager.performance_history.append(performance_record)
    
    # Keep only recent history to prevent memory bloat
    if len(manager.performance_history) > 100:
        manager.performance_history = manager.performance_history[-50:]


# Convenience functions for specific use cases
def flash_attention_v100_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """Convenience function for causal attention"""
    return flash_attention_v100(q, k, v, causal=True, **kwargs)


def flash_attention_v100_with_dropout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    **kwargs
) -> torch.Tensor:
    """Convenience function for attention with dropout"""
    return flash_attention_v100(q, k, v, dropout_p=dropout_p, **kwargs)


def flash_attention_v100_multi_gpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """Convenience function for multi-GPU attention"""
    return flash_attention_v100(q, k, v, enable_multi_gpu=True, **kwargs)


# Context managers for optimization
@contextlib.contextmanager
def flash_attention_performance_mode(strategy: str = 'speed'):
    """Context manager for temporary performance optimization"""
    global _global_manager
    old_manager = _global_manager
    
    try:
        _global_manager = FlashAttentionV100Manager(optimization_strategy=strategy)
        yield
    finally:
        _global_manager = old_manager


@contextlib.contextmanager  
def flash_attention_memory_mode():
    """Context manager for memory-optimized attention"""
    with flash_attention_performance_mode('memory'):
        yield


# Export main functions and classes
__all__ = [
    'flash_attention_v100',
    'flash_attention_v100_causal',
    'flash_attention_v100_with_dropout', 
    'flash_attention_v100_multi_gpu',
    'FlashAttentionV100Manager',
    'get_flash_attention_manager',
    'flash_attention_performance_mode',
    'flash_attention_memory_mode',
]