# utils.py
"""
Enhanced utility functions for Flash Attention V100

This module provides comprehensive utility functions with advanced features:
- Multi-GPU aware input validation and tensor management
- Enhanced numerical stability checks and corrections
- Performance monitoring and optimization utilities
- Advanced memory usage calculation and optimization
- Distributed computing support utilities
"""

import torch
import torch.distributed as dist
import math
import time
import warnings
import psutil
import os
from typing import Tuple, Optional, Union, Dict, List, Any
from functools import wraps
import contextlib

from .config import (
    SUPPORTED_DTYPES, NumericalConfig, V100MemoryHierarchy, 
    OptimizationConfig, DebugConfig, MultiGPUConfig
)


# Enhanced input validation with multi-GPU support
def validate_inputs_enhanced(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    enable_multi_gpu_checks: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Enhanced input validation with comprehensive checks
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len, num_heads, head_dim] 
        v: Value tensor [batch_size, seq_len, num_heads, head_dim]
        scale: Attention scale factor
        bias: Optional bias tensor
        dropout_p: Dropout probability
        enable_multi_gpu_checks: Whether to perform multi-GPU validation
        
    Returns:
        Validated and prepared q, k, v tensors and scale factor
    """
    
    # Basic tensor validation
    if not all(isinstance(t, torch.Tensor) for t in [q, k, v]):
        raise TypeError("q, k, v must be torch.Tensor instances")
    
    # Check tensor dimensions
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Input tensors must be 4-dimensional [batch, seq_len, num_heads, head_dim]")
    
    batch_size, seq_len_q, num_heads_q, head_dim_q = q.shape
    batch_size_k, seq_len_k, num_heads_k, head_dim_k = k.shape
    batch_size_v, seq_len_v, num_heads_v, head_dim_v = v.shape
    
    # Enhanced shape validation with detailed error messages
    if not (batch_size == batch_size_k == batch_size_v):
        raise ValueError(
            f"Batch sizes must match across Q, K, V. "
            f"Got Q: {batch_size}, K: {batch_size_k}, V: {batch_size_v}"
        )
    
    if seq_len_k != seq_len_v:
        raise ValueError(
            f"Sequence lengths of K and V must match. "
            f"Got K: {seq_len_k}, V: {seq_len_v}"
        )
    
    if not (head_dim_q == head_dim_k == head_dim_v):
        raise ValueError(
            f"Head dimensions must match across Q, K, V. "
            f"Got Q: {head_dim_q}, K: {head_dim_k}, V: {head_dim_v}"
        )
    
    # Enhanced head count validation (supports grouped attention)
    if num_heads_q % num_heads_k != 0 or num_heads_q % num_heads_v != 0:
        raise ValueError(
            f"Number of query heads ({num_heads_q}) must be divisible by "
            f"number of key heads ({num_heads_k}) and value heads ({num_heads_v})"
        )
    
    if num_heads_k != num_heads_v:
        raise ValueError(
            f"Number of key heads ({num_heads_k}) must equal number of value heads ({num_heads_v})"
        )
    
    # Data type validation with enhancement suggestions
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"All tensors must have the same dtype. "
            f"Got Q: {q.dtype}, K: {k.dtype}, V: {v.dtype}"
        )
    
    if q.dtype not in SUPPORTED_DTYPES:
        supported_list = ", ".join(str(dt) for dt in SUPPORTED_DTYPES)
        raise ValueError(
            f"Unsupported dtype {q.dtype}. Supported types: {supported_list}"
        )
    
    # Device validation with multi-GPU awareness
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"All tensors must be on the same device. "
            f"Got Q: {q.device}, K: {k.device}, V: {v.device}"
        )
    
    if not q.device.type == 'cuda':
        if torch.cuda.is_available():
            warnings.warn(
                f"Tensors are on {q.device.type} but CUDA is available. "
                "Consider moving tensors to GPU for better performance.",
                UserWarning
            )
        else:
            raise ValueError("Flash Attention V100 requires CUDA tensors")
    
    # Multi-GPU validation
    if enable_multi_gpu_checks and torch.cuda.device_count() > 1:
        _validate_multi_gpu_setup(q, k, v)
    
    # Bias validation
    if bias is not None:
        _validate_bias_tensor(bias, batch_size, num_heads_q, seq_len_q, seq_len_k)
    
    # Dropout validation
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
    
    # Numerical stability checks
    _check_numerical_stability(q, k, v)
    
    # Performance optimization suggestions
    _suggest_performance_optimizations(q, k, v)
    
    # Set default scale with enhanced precision
    if scale is None:
        scale = head_dim_q ** NumericalConfig.DEFAULT_SCALE_POWER
        
        # Adjust scale for numerical stability in extreme cases
        if head_dim_q > 128:
            scale *= NumericalConfig.ATTENTION_TEMPERATURE
    
    # Make tensors contiguous with optimal memory layout
    q = _optimize_tensor_layout(q)
    k = _optimize_tensor_layout(k)
    v = _optimize_tensor_layout(v)
    
    return q, k, v, scale


def _validate_multi_gpu_setup(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Validate multi-GPU setup and configuration"""
    
    current_device = q.device
    
    # Check if device supports V100+ features
    if current_device.type == 'cuda':
        capability = torch.cuda.get_device_capability(current_device)
        if capability < (7, 0):
            warnings.warn(
                f"Device {current_device} has compute capability {capability} < 7.0. "
                "Flash Attention V100 is optimized for V100+ GPUs.",
                UserWarning
            )
    
    # Check for distributed setup
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        if world_size > 1:
            # Validate tensor consistency across ranks
            batch_size = q.size(0)
            num_heads = q.size(2)
            
            if num_heads % world_size != 0:
                warnings.warn(
                    f"Number of heads ({num_heads}) is not evenly divisible by world size ({world_size}). "
                    "Head parallelization may not be optimal.",
                    UserWarning
                )


def _validate_bias_tensor(bias: torch.Tensor, batch_size: int, num_heads: int, 
                         seq_len_q: int, seq_len_k: int):
    """Validate bias tensor shape and properties"""
    
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a torch.Tensor")
    
    # Check bias shape compatibility
    expected_shapes = [
        (batch_size, num_heads, seq_len_q, seq_len_k),  # Full bias
        (1, num_heads, seq_len_q, seq_len_k),           # Broadcast batch
        (batch_size, 1, seq_len_q, seq_len_k),          # Broadcast heads
        (1, 1, seq_len_q, seq_len_k),                   # Broadcast both
        (num_heads, seq_len_q, seq_len_k),              # No batch dim
        (seq_len_q, seq_len_k),                         # Position bias only
    ]
    
    if bias.shape not in expected_shapes:
        shape_strs = [str(s) for s in expected_shapes]
        raise ValueError(
            f"bias shape {bias.shape} not compatible with attention shape. "
            f"Expected one of: {', '.join(shape_strs)}"
        )
    
    # Check for NaN/Inf in bias
    if torch.isnan(bias).any():
        raise ValueError("bias tensor contains NaN values")
    if torch.isinf(bias).any():
        raise ValueError("bias tensor contains Inf values")


def _check_numerical_stability(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Check numerical stability of input tensors"""
    
    for name, tensor in [("query", q), ("key", k), ("value", v)]:
        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} tensor contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} tensor contains Inf values")
        
        # Check for extreme values that might cause overflow
        max_val = tensor.abs().max().item()
        if max_val > 1e4:
            warnings.warn(
                f"{name} tensor has very large values (max: {max_val:.2e}). "
                "This may cause numerical instability.",
                UserWarning
            )
        
        # Check for very small values that might cause underflow
        if tensor.dtype in [torch.float16, torch.bfloat16]:
            min_val = tensor[tensor != 0].abs().min().item() if tensor.numel() > 0 else 1.0
            if min_val < 1e-7:
                warnings.warn(
                    f"{name} tensor has very small values (min: {min_val:.2e}) with dtype {tensor.dtype}. "
                    "Consider using float32 for better numerical stability.",
                    UserWarning
                )


def _suggest_performance_optimizations(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Suggest performance optimizations based on tensor properties"""
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Suggest Tensor Core optimization
    if head_dim % 16 != 0:
        nearest_16 = ((head_dim + 15) // 16) * 16
        warnings.warn(
            f"Head dimension {head_dim} is not divisible by 16. "
            f"Consider padding to {nearest_16} for optimal Tensor Core utilization.",
            UserWarning
        )
    
    if num_heads % 4 != 0:
        warnings.warn(
            f"Number of heads {num_heads} is not divisible by 4. "
            "This may lead to suboptimal memory access patterns.",
            UserWarning
        )
    
    # Memory usage suggestions
    estimated_memory = batch_size * seq_len * num_heads * head_dim * 4 * 6  # Rough estimate
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory
        if estimated_memory > available_memory * 0.8:
            warnings.warn(
                f"Estimated memory usage ({estimated_memory / 1024**3:.1f} GB) is high. "
                "Consider enabling gradient checkpointing or reducing batch size.",
                UserWarning
            )


def _optimize_tensor_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Optimize tensor memory layout for V100"""
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Ensure optimal alignment for Tensor Cores
    if tensor.data_ptr() % 256 != 0:  # 256-byte alignment
        # Create properly aligned tensor
        tensor = tensor.clone()
    
    return tensor


# Enhanced memory calculation with detailed breakdown
def calculate_memory_requirements_enhanced(
    batch_size: int, 
    seq_len: int, 
    num_heads: int, 
    head_dim: int, 
    dtype: torch.dtype,
    causal: bool = False,
    enable_optimizations: bool = True,
    num_gpus: int = 1,
    include_workspace: bool = True
) -> Dict[str, float]:
    """
    Enhanced memory requirements calculation with detailed breakdown
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        causal: Whether causal masking is used
        enable_optimizations: Whether optimizations are enabled
        num_gpus: Number of GPUs
        include_workspace: Whether to include workspace memory
        
    Returns:
        Dictionary with detailed memory requirement information
    """
    
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Input tensors memory (Q, K, V)
    single_tensor_size = batch_size * seq_len * num_heads * head_dim * element_size
    qkv_memory = 3 * single_tensor_size
    
    # Output tensor memory
    output_memory = single_tensor_size
    
    # Attention matrix memory (what Flash Attention avoids)
    full_attention_memory = batch_size * num_heads * seq_len * seq_len * element_size
    
    # Flash Attention block-wise memory (much smaller)
    from .config import BlockConfig
    if enable_optimizations:
        block_m = BlockConfig.DEFAULT_BLOCK_M
        block_n = BlockConfig.DEFAULT_BLOCK_N
    else:
        block_m = min(64, seq_len)
        block_n = min(64, seq_len)
    
    # Causal masking can reduce effective block size
    if causal:
        effective_block_size = min(block_m * block_n, seq_len * seq_len // 2)
    else:
        effective_block_size = block_m * block_n
    
    flash_intermediate_memory = batch_size * num_heads * effective_block_size * element_size
    
    # Forward pass auxiliary memory (LSE, max values)
    aux_memory = batch_size * num_heads * seq_len * 4 * 2  # fp32 for stability
    
    # Backward pass memory (gradients)
    gradient_memory = 3 * single_tensor_size  # dQ, dK, dV
    if not enable_optimizations:
        # Additional delta tensor if not fused
        gradient_memory += batch_size * num_heads * seq_len * 4
    
    # Workspace memory
    workspace_memory = 0
    if include_workspace:
        if enable_optimizations:
            # Optimized workspace with buffer reuse
            workspace_memory = max(flash_intermediate_memory, aux_memory) * 1.5
        else:
            # Conservative workspace allocation
            workspace_memory = (flash_intermediate_memory + aux_memory) * 2
    
    # Multi-GPU overhead
    multi_gpu_overhead = 0
    if num_gpus > 1:
        # Communication buffers and coordination overhead
        communication_memory = (qkv_memory + output_memory) * 0.2
        multi_gpu_overhead = communication_memory / num_gpus
    
    # Memory pool overhead
    pool_overhead = 0
    if OptimizationConfig.ENABLE_MEMORY_POOLING:
        total_active_memory = qkv_memory + output_memory + flash_intermediate_memory
        pool_overhead = total_active_memory * 0.15  # 15% overhead for pooling
    
    # Total calculations
    forward_memory = qkv_memory + output_memory + flash_intermediate_memory + aux_memory
    total_training_memory = forward_memory + gradient_memory + workspace_memory
    total_inference_memory = forward_memory + workspace_memory
    
    # Add overheads
    total_training_memory += multi_gpu_overhead + pool_overhead
    total_inference_memory += multi_gpu_overhead + pool_overhead
    
    # Memory savings calculation
    naive_training_memory = qkv_memory + output_memory + full_attention_memory + gradient_memory
    memory_savings_ratio = naive_training_memory / total_training_memory if total_training_memory > 0 else 1.0
    
    # Efficiency metrics
    theoretical_minimum = qkv_memory + output_memory + gradient_memory
    efficiency_ratio = theoretical_minimum / total_training_memory if total_training_memory > 0 else 1.0
    
    return {
        # Core memory components (MB)
        'qkv_memory_mb': qkv_memory / (1024**2),
        'output_memory_mb': output_memory / (1024**2),
        'full_attention_memory_mb': full_attention_memory / (1024**2),
        'flash_intermediate_memory_mb': flash_intermediate_memory / (1024**2),
        'auxiliary_memory_mb': aux_memory / (1024**2),
        'gradient_memory_mb': gradient_memory / (1024**2),
        'workspace_memory_mb': workspace_memory / (1024**2),
        
        # Overhead components (MB)
        'multi_gpu_overhead_mb': multi_gpu_overhead / (1024**2),
        'pool_overhead_mb': pool_overhead / (1024**2),
        
        # Total memory requirements (MB)
        'total_training_memory_mb': total_training_memory / (1024**2),
        'total_inference_memory_mb': total_inference_memory / (1024**2),
        'naive_training_memory_mb': naive_training_memory / (1024**2),
        
        # Efficiency metrics
        'memory_savings_ratio': memory_savings_ratio,
        'efficiency_ratio': efficiency_ratio,
        'overhead_percentage': ((multi_gpu_overhead + pool_overhead) / total_training_memory * 100) if total_training_memory > 0 else 0,
        
        # Configuration info
        'block_size': (block_m, block_n),
        'effective_block_elements': effective_block_size,
        'element_size_bytes': element_size,
        'optimizations_enabled': enable_optimizations,
        'num_gpus': num_gpus,
    }


# Performance monitoring and profiling utilities
class PerformanceProfiler:
    """Performance profiler for Flash Attention operations"""
    
    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.reset()
    
    def reset(self):
        """Reset profiling statistics"""
        self.operation_times = []
        self.memory_snapshots = []
        self.kernel_times = {}
        self.total_operations = 0
        self.start_time = None
    
    @contextlib.contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling individual operations"""
        
        start_time = time.perf_counter()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            operation_time = (end_time - start_time) * 1000  # Convert to ms
            memory_delta = final_memory - initial_memory
            
            self.operation_times.append({
                'name': operation_name,
                'time_ms': operation_time,
                'memory_delta_mb': memory_delta / (1024**2),
                'timestamp': end_time
            })
            
            self.total_operations += 1
            
            if operation_name not in self.kernel_times:
                self.kernel_times[operation_name] = []
            self.kernel_times[operation_name].append(operation_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance profiling summary"""
        
        if not self.operation_times:
            return {"error": "No operations profiled"}
        
        total_time = sum(op['time_ms'] for op in self.operation_times)
        total_memory = sum(op['memory_delta_mb'] for op in self.operation_times)
        
        # Per-operation statistics
        operation_stats = {}
        for op_name, times in self.kernel_times.items():
            operation_stats[op_name] = {
                'count': len(times),
                'total_time_ms': sum(times),
                'avg_time_ms': sum(times) / len(times) if times else 0,
                'min_time_ms': min(times) if times else 0,
                'max_time_ms': max(times) if times else 0,
            }
        
        return {
            'total_operations': self.total_operations,
            'total_time_ms': total_time,
            'avg_operation_time_ms': total_time / self.total_operations if self.total_operations > 0 else 0,
            'total_memory_delta_mb': total_memory,
            'operation_statistics': operation_stats,
            'timeline': self.operation_times if self.enable_detailed_profiling else None,
        }


# Distributed computing utilities
def setup_distributed_environment():
    """Setup distributed environment for multi-GPU training"""
    
    if dist.is_initialized():
        return
    
    # Check for SLURM environment
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        node_id = int(os.environ['SLURM_NODEID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    # Check for standard distributed environment
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single-node multi-GPU setup
        rank = 0
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        local_rank = 0
    
    if world_size > 1:
        try:
            # Initialize process group
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=rank,
                world_size=world_size
            )
            
            # Set current device for CUDA
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            print(f"Distributed environment initialized: rank {rank}/{world_size}")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize distributed environment: {e}", UserWarning)


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
        'distributed_available': dist.is_available(),
        'distributed_initialized': dist.is_initialized() if dist.is_available() else False,
    }
    
    if torch.cuda.is_available():
        device_details = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capability = torch.cuda.get_device_capability(i)
            
            device_details.append({
                'device_id': i,
                'name': props.name,
                'compute_capability': capability,
                'total_memory_gb': props.total_memory / (1024**3),
                'multi_processor_count': props.multi_processor_count,
                'v100_compatible': capability >= (7, 0),
                'tensor_core_support': capability >= (7, 0),
            })
        
        device_info['devices'] = device_details
        device_info['current_device'] = torch.cuda.current_device()
    
    # System information
    try:
        device_info['system'] = {
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3) if hasattr(psutil, 'virtual_memory') else None,
        }
    except:
        pass
    
    return device_info


# Tensor manipulation utilities
def cdiv(a: int, b: int) -> int:
    """Ceiling division"""
    return (a + b - 1) // b


def next_power_of_2(x: int) -> int:
    """Get the next power of 2 greater than or equal to x"""
    return 2 ** math.ceil(math.log2(max(1, x)))


def is_power_of_2(x: int) -> bool:
    """Check if x is a power of 2"""
    return x > 0 and (x & (x - 1)) == 0


def pad_to_multiple(x: int, multiple: int) -> int:
    """Pad x to the next multiple"""
    return cdiv(x, multiple) * multiple


def optimize_tensor_for_v100(tensor: torch.Tensor) -> torch.Tensor:
    """Optimize tensor layout and properties for V100"""
    
    # Ensure contiguous memory layout
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Check data type optimization
    if tensor.dtype == torch.float32 and tensor.requires_grad:
        warnings.warn(
            "Using float32 with gradients. Consider float16 with loss scaling for better V100 performance.",
            UserWarning
        )
    
    # Check shape optimization for Tensor Cores
    if len(tensor.shape) >= 2:
        last_two_dims = tensor.shape[-2:]
        if any(dim % 16 != 0 for dim in last_two_dims):
            warnings.warn(
                f"Tensor dimensions {last_two_dims} are not optimal for Tensor Cores. "
                "Consider padding to multiples of 16.",
                UserWarning
            )
    
    return tensor


def log_memory_usage(tensor_dict: Dict[str, torch.Tensor], prefix: str = ""):
    """Enhanced memory usage logging"""
    
    if not DebugConfig.LOG_MEMORY_USAGE:
        return
    
    print(f"{prefix}Enhanced Memory Usage Report:")
    print("-" * 50)
    
    total_memory = 0
    total_elements = 0
    
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            total_memory += memory_mb
            total_elements += tensor.numel()
            
            print(f"  {name:20s}: {memory_mb:8.2f} MB | {tensor.shape} | {tensor.dtype} | {tensor.device}")
    
    print("-" * 50)
    print(f"  {'TOTAL':20s}: {total_memory:8.2f} MB | {total_elements:,} elements")
    
    # GPU memory info if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        cached = torch.cuda.memory_reserved() / (1024**2)
        print(f"  {'GPU Allocated':20s}: {allocated:8.2f} MB")
        print(f"  {'GPU Cached':20s}: {cached:8.2f} MB")
    
    print("-" * 50)


def check_tensor_nan_inf(tensor: torch.Tensor, name: str = "tensor", 
                        fix_issues: bool = False) -> torch.Tensor:
    """Enhanced NaN/Inf checking with optional fixing"""
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        issue_str = []
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            issue_str.append(f"{nan_count} NaN values")
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            issue_str.append(f"{inf_count} Inf values")
        
        error_msg = f"{name} contains {' and '.join(issue_str)}"
        
        if fix_issues:
            warnings.warn(f"{error_msg}. Attempting to fix...", UserWarning)
            
            # Replace NaN with zeros
            if has_nan:
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            
            # Clamp infinite values
            if has_inf:
                max_finite = torch.finfo(tensor.dtype).max / 2
                tensor = torch.clamp(tensor, -max_finite, max_finite)
            
            return tensor
        else:
            raise ValueError(error_msg)
    
    return tensor


# Debugging and visualization utilities
def create_attention_visualization_data(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None,
    seq_len_limit: int = 100
) -> Dict[str, Any]:
    """Create data for attention pattern visualization"""
    
    if attention_weights is None:
        # Compute attention weights for visualization
        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(scores, dim=-1)
    
    # Limit sequence length for visualization
    seq_len = min(attention_weights.size(-1), seq_len_limit)
    vis_weights = attention_weights[0, 0, :seq_len, :seq_len]  # First batch, first head
    
    return {
        'attention_matrix': vis_weights.detach().cpu().numpy(),
        'query_norms': torch.norm(q[0, :seq_len, 0, :], dim=-1).detach().cpu().numpy(),
        'key_norms': torch.norm(k[0, :seq_len, 0, :], dim=-1).detach().cpu().numpy(),
        'sequence_length': seq_len,
        'head_dimension': q.size(-1),
        'attention_entropy': -torch.sum(vis_weights * torch.log(vis_weights + 1e-9), dim=-1).mean().item(),
    }


# Performance benchmarking utilities
def benchmark_operation(
    operation_func,
    *args,
    num_warmup: int = 10,
    num_runs: int = 100,
    return_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Comprehensive operation benchmarking"""
    
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available for benchmarking", UserWarning)
        return {}
    
    # Warmup runs
    for _ in range(num_warmup):
        try:
            _ = operation_func(*args, **kwargs)
        except Exception as e:
            return {'error': f"Warmup failed: {str(e)}"}
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    times = []
    output = None
    
    start_memory = torch.cuda.memory_allocated()
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        try:
            result = operation_func(*args, **kwargs)
            if return_output and i == 0:
                output = result
        except Exception as e:
            return {'error': f"Benchmark run {i} failed: {str(e)}"}
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Calculate statistics
    import numpy as np
    times = np.array(times)
    
    result = {
        'avg_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
        'memory_delta_mb': (end_memory - start_memory) / (1024**2),
        'peak_memory_mb': peak_memory / (1024**2),
        'num_runs': num_runs,
        'num_warmup': num_warmup,
    }
    
    if return_output:
        result['output'] = output
    
    return result


# Export utility functions
__all__ = [
    'validate_inputs_enhanced',
    'calculate_memory_requirements_enhanced',
    'PerformanceProfiler',
    'setup_distributed_environment',
    'get_device_info',
    'cdiv',
    'next_power_of_2',
    'is_power_of_2',
    'pad_to_multiple',
    'optimize_tensor_for_v100',
    'log_memory_usage',
    'check_tensor_nan_inf',
    'create_attention_visualization_data',
    'benchmark_operation',
    
    # Legacy compatibility
    'validate_inputs',
    'calculate_memory_requirements',
    'get_tensor_info',
    'get_optimal_block_size',
]

# Provide aliases for backward compatibility
validate_inputs = validate_inputs_enhanced
calculate_memory_requirements = calculate_memory_requirements_enhanced

def get_tensor_info(tensor: torch.Tensor) -> dict:
    """Legacy compatibility function"""
    return {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'is_contiguous': tensor.is_contiguous(),
        'memory_usage_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
        'requires_grad': tensor.requires_grad
    }

def get_optimal_block_size(seq_len: int, head_dim: int, available_memory: int) -> Tuple[int, int]:
    """Legacy compatibility function"""
    from .config import get_enhanced_config
    config = get_enhanced_config(seq_len, head_dim)
    return config['BLOCK_M'], config['BLOCK_N']