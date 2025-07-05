# memory.py
"""
Memory management for Flash Attention V100

This module provides memory pool management, workspace allocation,
and memory usage optimization for Flash Attention operations.
"""

import torch
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import weakref

from ..config import V100MemoryHierarchy, OptimizationConfig
from ..utils import calculate_memory_requirements


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool"""
    tensor: torch.Tensor
    size_bytes: int
    in_use: bool = False
    created_at: float = 0.0
    last_used: float = 0.0


@dataclass
class MemoryStatistics:
    """Memory usage statistics"""
    total_allocated_mb: float
    peak_allocated_mb: float
    total_cached_mb: float
    active_blocks: int
    free_blocks: int
    pool_efficiency: float  # Fraction of pool memory in use


class MemoryManager:
    """
    Memory pool manager for Flash Attention operations
    
    This class manages memory allocation and reuse to minimize
    GPU memory allocation overhead and fragmentation.
    """
    
    def __init__(self, max_pool_size_gb: float = 2.0):
        """
        Initialize memory manager
        
        Args:
            max_pool_size_gb: Maximum memory pool size in GB
        """
        self.max_pool_size = int(max_pool_size_gb * 1024**3)  # Convert to bytes
        self.memory_pools: Dict[torch.device, Dict[torch.dtype, List[MemoryBlock]]] = {}
        self.allocated_tensors: weakref.WeakSet = weakref.WeakSet()
        self.peak_memory_usage = 0
        self.total_allocations = 0
        self.total_deallocations = 0
        
        # Enable memory pooling if configured
        self.enable_pooling = OptimizationConfig.ENABLE_MEMORY_POOLING
    
    def allocate(
        self, 
        shape: Tuple[int, ...], 
        dtype: torch.dtype, 
        device: torch.device,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Allocate tensor from memory pool or create new one
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Allocated tensor
        """
        
        if not self.enable_pooling:
            return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        
        size_bytes = torch.empty(shape, dtype=dtype).numel() * torch.tensor(0, dtype=dtype).element_size()
        
        # Try to find suitable block in pool
        suitable_block = self._find_suitable_block(device, dtype, size_bytes)
        
        if suitable_block is not None:
            # Reuse existing block
            suitable_block.in_use = True
            suitable_block.last_used = torch.cuda.Event().query()
            
            # Reshape if necessary
            tensor = suitable_block.tensor.view(shape)
            tensor.requires_grad_(requires_grad)
            
            self.allocated_tensors.add(tensor)
            return tensor
        
        # Create new tensor
        tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        
        # Add to pool if under size limit
        current_pool_size = self._get_pool_size(device, dtype)
        if current_pool_size + size_bytes <= self.max_pool_size:
            self._add_to_pool(tensor, device, dtype, size_bytes)
        
        self.allocated_tensors.add(tensor)
        self.total_allocations += 1
        
        # Update peak memory usage
        current_usage = torch.cuda.memory_allocated(device)
        self.peak_memory_usage = max(self.peak_memory_usage, current_usage)
        
        return tensor
    
    def deallocate(self, tensor: torch.Tensor):
        """
        Mark tensor as available for reuse
        
        Args:
            tensor: Tensor to deallocate
        """
        
        if not self.enable_pooling:
            return
        
        device = tensor.device
        dtype = tensor.dtype
        
        # Find corresponding block in pool
        if device in self.memory_pools and dtype in self.memory_pools[device]:
            for block in self.memory_pools[device][dtype]:
                if torch.equal(block.tensor.storage().data_ptr(), tensor.storage().data_ptr()):
                    block.in_use = False
                    self.total_deallocations += 1
                    return
    
    def _find_suitable_block(
        self, 
        device: torch.device, 
        dtype: torch.dtype, 
        size_bytes: int
    ) -> Optional[MemoryBlock]:
        """Find suitable memory block from pool"""
        
        if device not in self.memory_pools:
            return None
        
        if dtype not in self.memory_pools[device]:
            return None
        
        # Find smallest block that fits the requirement
        suitable_blocks = [
            block for block in self.memory_pools[device][dtype]
            if not block.in_use and block.size_bytes >= size_bytes
        ]
        
        if not suitable_blocks:
            return None
        
        # Return smallest suitable block to minimize waste
        return min(suitable_blocks, key=lambda b: b.size_bytes)
    
    def _add_to_pool(
        self, 
        tensor: torch.Tensor, 
        device: torch.device, 
        dtype: torch.dtype, 
        size_bytes: int
    ):
        """Add tensor to memory pool"""
        
        if device not in self.memory_pools:
            self.memory_pools[device] = {}
        
        if dtype not in self.memory_pools[device]:
            self.memory_pools[device][dtype] = []
        
        block = MemoryBlock(
            tensor=tensor,
            size_bytes=size_bytes,
            in_use=True,
            created_at=torch.cuda.Event().query(),
            last_used=torch.cuda.Event().query()
        )
        
        self.memory_pools[device][dtype].append(block)
    
    def _get_pool_size(self, device: torch.device, dtype: torch.dtype) -> int:
        """Get current pool size for device and dtype"""
        
        if device not in self.memory_pools:
            return 0
        
        if dtype not in self.memory_pools[device]:
            return 0
        
        return sum(block.size_bytes for block in self.memory_pools[device][dtype])
    
    def cleanup(self, force: bool = False):
        """
        Clean up memory pool
        
        Args:
            force: If True, free all memory regardless of usage
        """
        
        for device_pools in self.memory_pools.values():
            for dtype_pool in device_pools.values():
                blocks_to_remove = []
                
                for i, block in enumerate(dtype_pool):
                    if force or not block.in_use:
                        blocks_to_remove.append(i)
                
                # Remove blocks in reverse order to maintain indices
                for i in reversed(blocks_to_remove):
                    del dtype_pool[i]
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_statistics(self) -> MemoryStatistics:
        """Get memory usage statistics"""
        
        total_allocated = 0
        total_cached = 0
        active_blocks = 0
        free_blocks = 0
        
        for device_pools in self.memory_pools.values():
            for dtype_pool in device_pools.values():
                for block in dtype_pool:
                    total_cached += block.size_bytes
                    if block.in_use:
                        total_allocated += block.size_bytes
                        active_blocks += 1
                    else:
                        free_blocks += 1
        
        pool_efficiency = total_allocated / total_cached if total_cached > 0 else 0
        
        return MemoryStatistics(
            total_allocated_mb=total_allocated / (1024**2),
            peak_allocated_mb=self.peak_memory_usage / (1024**2),
            total_cached_mb=total_cached / (1024**2),
            active_blocks=active_blocks,
            free_blocks=free_blocks,
            pool_efficiency=pool_efficiency
        )


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def allocate_attention_workspace(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int, 
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    include_backward: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Allocate workspace tensors for Flash Attention
    
    Args:
        batch_size: Batch size
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        device: Device
        include_backward: Whether to allocate tensors for backward pass
        
    Returns:
        Dictionary with allocated workspace tensors
    """
    
    manager = get_memory_manager()
    workspace = {}
    
    # Forward pass tensors
    workspace['output'] = manager.allocate(
        (batch_size, seq_len_q, num_heads, head_dim), dtype, device
    )
    
    workspace['lse'] = manager.allocate(
        (batch_size, num_heads, seq_len_q), torch.float32, device
    )
    
    workspace['max_vals'] = manager.allocate(
        (batch_size, num_heads, seq_len_q), torch.float32, device
    )
    
    # Backward pass tensors
    if include_backward:
        workspace['grad_q'] = manager.allocate(
            (batch_size, seq_len_q, num_heads, head_dim), dtype, device
        )
        
        workspace['grad_k'] = manager.allocate(
            (batch_size, seq_len_k, num_heads, head_dim), dtype, device
        )
        
        workspace['grad_v'] = manager.allocate(
            (batch_size, seq_len_k, num_heads, head_dim), dtype, device
        )
        
        workspace['delta'] = manager.allocate(
            (batch_size, num_heads, seq_len_q), torch.float32, device
        )
    
    return workspace


def free_attention_workspace(workspace: Dict[str, torch.Tensor]):
    """
    Free workspace tensors
    
    Args:
        workspace: Dictionary with workspace tensors
    """
    
    manager = get_memory_manager()
    
    for tensor in workspace.values():
        if isinstance(tensor, torch.Tensor):
            manager.deallocate(tensor)


def estimate_peak_memory_usage(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    include_gradients: bool = True
) -> Dict[str, float]:
    """
    Estimate peak memory usage for Flash Attention
    
    Args:
        batch_size: Batch size
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        include_gradients: Whether to include gradient memory
        
    Returns:
        Dictionary with memory usage estimates in MB
    """
    
    memory_info = calculate_memory_requirements(
        batch_size, max(seq_len_q, seq_len_k), num_heads, head_dim, dtype
    )
    
    # Input tensors (Q, K, V)
    input_memory = memory_info['qkv_memory_mb']
    
    # Output tensor
    output_memory = memory_info['output_memory_mb']
    
    # Flash Attention intermediate memory
    intermediate_memory = memory_info['flash_intermediate_memory_mb']
    
    # Forward pass statistics (LSE, max values)
    stats_memory = batch_size * num_heads * seq_len_q * 4 * 2 / (1024**2)  # fp32
    
    # Gradient memory (if training)
    gradient_memory = 0
    if include_gradients:
        gradient_memory = input_memory + output_memory  # dQ, dK, dV, dO
    
    # Workspace memory
    workspace_memory = stats_memory * 2  # Additional workspace for computations
    
    total_memory = (
        input_memory + output_memory + intermediate_memory + 
        stats_memory + gradient_memory + workspace_memory
    )
    
    return {
        'input_memory_mb': input_memory,
        'output_memory_mb': output_memory,
        'intermediate_memory_mb': intermediate_memory,
        'statistics_memory_mb': stats_memory,
        'gradient_memory_mb': gradient_memory,
        'workspace_memory_mb': workspace_memory,
        'total_memory_mb': total_memory,
        'memory_savings_ratio': memory_info['memory_savings_ratio'],
    }


def get_memory_statistics() -> Dict[str, Any]:
    """
    Get comprehensive memory statistics
    
    Returns:
        Dictionary with memory statistics
    """
    
    # PyTorch CUDA memory stats
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda_stats = torch.cuda.memory_stats(device)
        allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
        cached_mb = torch.cuda.memory_reserved(device) / (1024**2)
        max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        max_cached_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        cuda_stats = {}
        allocated_mb = cached_mb = max_allocated_mb = max_cached_mb = 0
    
    # Memory manager stats
    manager = get_memory_manager()
    pool_stats = manager.get_statistics()
    
    return {
        'cuda_memory': {
            'allocated_mb': allocated_mb,
            'cached_mb': cached_mb,
            'max_allocated_mb': max_allocated_mb,
            'max_cached_mb': max_cached_mb,
            'raw_stats': cuda_stats,
        },
        'memory_pool': {
            'total_allocated_mb': pool_stats.total_allocated_mb,
            'peak_allocated_mb': pool_stats.peak_allocated_mb,
            'total_cached_mb': pool_stats.total_cached_mb,
            'active_blocks': pool_stats.active_blocks,
            'free_blocks': pool_stats.free_blocks,
            'pool_efficiency': pool_stats.pool_efficiency,
        },
        'allocations': {
            'total_allocations': manager.total_allocations,
            'total_deallocations': manager.total_deallocations,
        }
    }


def cleanup_memory():
    """Clean up all memory pools and caches"""
    manager = get_memory_manager()
    manager.cleanup(force=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()


# Export memory management functions
__all__ = [
    'MemoryManager',
    'MemoryBlock',
    'MemoryStatistics',
    'allocate_attention_workspace',
    'free_attention_workspace',
    'estimate_peak_memory_usage',
    'get_memory_statistics',
    'cleanup_memory',
    'get_memory_manager',
]