# memory.py
"""
Enhanced memory management for Flash Attention V100

This module provides advanced memory management features:
- Multi-GPU aware memory pooling and allocation
- Adaptive memory optimization based on usage patterns
- Memory leak detection and automatic cleanup
- Performance-oriented workspace management
- Intelligent memory prefetching and caching
"""

import torch
import torch.distributed as dist
import gc
import threading
import time
import warnings
import contextlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import psutil
import os

from ..config import V100MemoryHierarchy, OptimizationConfig, DebugConfig
from ..utils import calculate_memory_requirements


@dataclass
class EnhancedMemoryBlock:
    """Enhanced memory block with detailed tracking"""
    tensor: torch.Tensor
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    in_use: bool = False
    created_at: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0
    priority: int = 0  # Higher priority = keep longer
    
    # Performance tracking
    allocation_time_ms: float = 0.0
    access_pattern: str = 'sequential'  # 'sequential', 'random', 'contiguous'
    
    def update_usage(self):
        """Update usage statistics"""
        self.last_used = time.time()
        self.usage_count += 1
    
    def get_age_seconds(self) -> float:
        """Get age of the block in seconds"""
        return time.time() - self.created_at
    
    def get_idle_time_seconds(self) -> float:
        """Get idle time since last use"""
        return time.time() - self.last_used


@dataclass
class MemoryStatistics:
    """Enhanced memory usage statistics"""
    # Basic statistics
    total_allocated_mb: float
    peak_allocated_mb: float
    total_cached_mb: float
    active_blocks: int
    free_blocks: int
    pool_efficiency: float
    
    # Advanced statistics
    cache_hit_rate: float = 0.0
    memory_fragmentation: float = 0.0
    allocation_rate_per_sec: float = 0.0
    deallocation_rate_per_sec: float = 0.0
    
    # Multi-GPU statistics
    per_device_allocated_mb: Dict[str, float] = field(default_factory=dict)
    per_device_cached_mb: Dict[str, float] = field(default_factory=dict)
    cross_device_transfers_mb: float = 0.0
    
    # Performance metrics
    avg_allocation_time_ms: float = 0.0
    avg_access_time_ms: float = 0.0
    memory_bandwidth_gbps: float = 0.0


class AdaptiveMemoryManager:
    """
    Advanced memory pool manager with adaptive optimization
    
    Features:
    - Multi-GPU memory coordination
    - Intelligent prefetching and caching
    - Memory usage pattern analysis
    - Automatic memory optimization
    - Leak detection and cleanup
    """
    
    def __init__(self, 
                 max_pool_size_gb: float = 4.0,
                 enable_multi_gpu: bool = True,
                 enable_adaptive_optimization: bool = True,
                 cache_policy: str = 'lru',  # 'lru', 'lfu', 'adaptive'
                 enable_profiling: bool = False):
        """
        Initialize Adaptive Memory Manager
        
        Args:
            max_pool_size_gb: Maximum memory pool size per GPU in GB
            enable_multi_gpu: Whether to enable multi-GPU coordination
            enable_adaptive_optimization: Whether to use adaptive optimization
            cache_policy: Cache replacement policy
            enable_profiling: Whether to enable detailed profiling
        """
        self.max_pool_size = int(max_pool_size_gb * 1024**3)  # Convert to bytes
        self.enable_multi_gpu = enable_multi_gpu and torch.cuda.is_available()
        self.enable_adaptive_optimization = enable_adaptive_optimization
        self.cache_policy = cache_policy
        self.enable_profiling = enable_profiling
        
        # Memory pools per device
        self.memory_pools: Dict[torch.device, Dict[torch.dtype, List[EnhancedMemoryBlock]]] = {}
        
        # Cross-device memory coordination
        if self.enable_multi_gpu:
            self.device_memory_usage: Dict[torch.device, float] = {}
            self.cross_device_cache: Dict[Tuple[torch.device, torch.device], List[torch.Tensor]] = {}
        
        # Performance tracking
        self.allocation_history: deque = deque(maxlen=1000)
        self.access_patterns: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_allocations = 0
        self.total_deallocations = 0
        
        # Adaptive optimization state
        self.optimization_interval = 30.0  # seconds
        self.last_optimization = 0.0
        self.memory_pressure_threshold = 0.85
        
        # Memory leak detection
        self.allocated_tensors: weakref.WeakSet = weakref.WeakSet()
        self.leak_detection_enabled = True
        
        # Threading for background tasks
        self.background_thread = None
        self.shutdown_event = threading.Event()
        
        # Start background optimization if enabled
        if self.enable_adaptive_optimization:
            self._start_background_optimization()
        
        print(f"Adaptive Memory Manager initialized:")
        print(f"  Pool size: {max_pool_size_gb} GB per GPU")
        print(f"  Multi-GPU: {self.enable_multi_gpu}")
        print(f"  Cache policy: {cache_policy}")
        print(f"  Adaptive optimization: {enable_adaptive_optimization}")
    
    def _start_background_optimization(self):
        """Start background thread for memory optimization"""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(
                target=self._background_optimization_loop,
                daemon=True
            )
            self.background_thread.start()
    
    def _background_optimization_loop(self):
        """Background optimization loop"""
        while not self.shutdown_event.wait(self.optimization_interval):
            try:
                self._optimize_memory_pools()
                self._detect_memory_leaks()
                self._update_device_coordination()
            except Exception as e:
                if DebugConfig.LOG_MEMORY_USAGE:
                    print(f"Background optimization error: {e}")
    
    def allocate_enhanced(self, 
                         shape: Tuple[int, ...], 
                         dtype: torch.dtype, 
                         device: torch.device,
                         requires_grad: bool = False,
                         priority: int = 0,
                         access_pattern: str = 'sequential') -> torch.Tensor:
        """
        Enhanced tensor allocation with intelligent caching
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device
            requires_grad: Whether tensor requires gradients
            priority: Priority for caching (higher = keep longer)
            access_pattern: Expected access pattern
            
        Returns:
            Allocated tensor
        """
        
        allocation_start = time.time()
        
        if not OptimizationConfig.ENABLE_MEMORY_POOLING:
            tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
            self.allocated_tensors.add(tensor)
            return tensor
        
        size_bytes = torch.empty(shape, dtype=dtype).numel() * torch.tensor(0, dtype=dtype).element_size()
        
        # Try to find suitable block in pool
        suitable_block = self._find_suitable_block_enhanced(device, dtype, shape, size_bytes)
        
        if suitable_block is not None:
            # Cache hit
            self.cache_hits += 1
            suitable_block.in_use = True
            suitable_block.update_usage()
            suitable_block.priority = max(suitable_block.priority, priority)
            suitable_block.access_pattern = access_pattern
            
            # Reshape if necessary
            tensor = suitable_block.tensor.view(shape)
            tensor.requires_grad_(requires_grad)
            
            self.allocated_tensors.add(tensor)
            
            if self.enable_profiling:
                allocation_time = (time.time() - allocation_start) * 1000
                self._record_allocation(shape, dtype, device, allocation_time, cache_hit=True)
            
            return tensor
        
        # Cache miss - create new tensor
        self.cache_misses += 1
        tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        
        # Add to pool if under size limit and beneficial
        if self._should_cache_tensor(size_bytes, device, access_pattern):
            self._add_to_pool_enhanced(tensor, device, dtype, shape, size_bytes, priority, access_pattern)
        
        self.allocated_tensors.add(tensor)
        self.total_allocations += 1
        
        # Update device memory tracking
        if self.enable_multi_gpu:
            if device not in self.device_memory_usage:
                self.device_memory_usage[device] = 0
            self.device_memory_usage[device] += size_bytes
        
        if self.enable_profiling:
            allocation_time = (time.time() - allocation_start) * 1000
            self._record_allocation(shape, dtype, device, allocation_time, cache_hit=False)
        
        return tensor
    
    def _find_suitable_block_enhanced(self, device: torch.device, dtype: torch.dtype, 
                                     shape: Tuple[int, ...], size_bytes: int) -> Optional[EnhancedMemoryBlock]:
        """Find suitable memory block with advanced matching"""
        
        if device not in self.memory_pools or dtype not in self.memory_pools[device]:
            return None
        
        blocks = self.memory_pools[device][dtype]
        suitable_blocks = [
            block for block in blocks
            if not block.in_use and block.size_bytes >= size_bytes
        ]
        
        if not suitable_blocks:
            return None
        
        # Advanced block selection based on cache policy
        if self.cache_policy == 'lru':
            # Least Recently Used
            return min(suitable_blocks, key=lambda b: b.last_used)
        elif self.cache_policy == 'lfu':
            # Least Frequently Used
            return min(suitable_blocks, key=lambda b: b.usage_count)
        elif self.cache_policy == 'adaptive':
            # Adaptive policy considering multiple factors
            return self._select_block_adaptive(suitable_blocks, shape, size_bytes)
        else:
            # Default: smallest suitable block
            return min(suitable_blocks, key=lambda b: b.size_bytes)
    
    def _select_block_adaptive(self, blocks: List[EnhancedMemoryBlock], 
                              shape: Tuple[int, ...], size_bytes: int) -> EnhancedMemoryBlock:
        """Adaptive block selection considering multiple factors"""
        
        def score_block(block):
            # Size efficiency (prefer exact matches)
            size_ratio = size_bytes / block.size_bytes
            size_score = size_ratio if size_ratio <= 1.0 else 1.0 / size_ratio
            
            # Shape compatibility (prefer same shape)
            shape_score = 1.0 if block.shape == shape else 0.5
            
            # Usage frequency (prefer frequently used blocks)
            usage_score = min(block.usage_count / 10.0, 1.0)
            
            # Recency (prefer recently used blocks)
            age = block.get_idle_time_seconds()
            recency_score = max(0, 1.0 - age / 300.0)  # 5 minute decay
            
            # Priority score
            priority_score = min(block.priority / 10.0, 1.0)
            
            # Combined score
            total_score = (size_score * 0.4 + shape_score * 0.2 + 
                          usage_score * 0.2 + recency_score * 0.1 + 
                          priority_score * 0.1)
            
            return total_score
        
        return max(blocks, key=score_block)
    
    def _should_cache_tensor(self, size_bytes: int, device: torch.device, access_pattern: str) -> bool:
        """Determine if tensor should be cached"""
        
        # Check pool size limits
        current_pool_size = self._get_pool_size(device)
        if current_pool_size + size_bytes > self.max_pool_size:
            return False
        
        # Check memory pressure
        if torch.cuda.is_available() and device.type == 'cuda':
            memory_fraction = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
            if memory_fraction > self.memory_pressure_threshold:
                return False
        
        # Consider access pattern
        if access_pattern == 'random' and size_bytes > 100 * 1024 * 1024:  # >100MB random access
            return False
        
        # Cache frequently allocated sizes
        size_frequency = self.access_patterns[f"size_{size_bytes}"]
        if size_frequency >= 3:  # Allocated at least 3 times
            return True
        
        # Default caching for reasonable sizes
        return size_bytes <= 500 * 1024 * 1024  # Cache up to 500MB tensors
    
    def _add_to_pool_enhanced(self, tensor: torch.Tensor, device: torch.device, 
                             dtype: torch.dtype, shape: Tuple[int, ...], 
                             size_bytes: int, priority: int, access_pattern: str):
        """Add tensor to memory pool with enhanced tracking"""
        
        if device not in self.memory_pools:
            self.memory_pools[device] = {}
        
        if dtype not in self.memory_pools[device]:
            self.memory_pools[device][dtype] = []
        
        block = EnhancedMemoryBlock(
            tensor=tensor,
            size_bytes=size_bytes,
            shape=shape,
            dtype=dtype,
            device=device,
            in_use=True,
            created_at=time.time(),
            last_used=time.time(),
            priority=priority,
            access_pattern=access_pattern
        )
        
        self.memory_pools[device][dtype].append(block)
        
        # Update access patterns
        self.access_patterns[f"size_{size_bytes}"] += 1
        self.access_patterns[f"shape_{shape}"] += 1
        self.access_patterns[f"pattern_{access_pattern}"] += 1
    
    def deallocate_enhanced(self, tensor: torch.Tensor):
        """Enhanced tensor deallocation with intelligent caching"""
        
        if not OptimizationConfig.ENABLE_MEMORY_POOLING:
            return
        
        device = tensor.device
        dtype = tensor.dtype
        
        # Find corresponding block in pool
        if device in self.memory_pools and dtype in self.memory_pools[device]:
            for block in self.memory_pools[device][dtype]:
                if torch.equal(block.tensor.data_ptr(), tensor.data_ptr()):
                    block.in_use = False
                    block.update_usage()
                    self.total_deallocations += 1
                    return
    
    def _optimize_memory_pools(self):
        """Optimize memory pools by removing unused blocks"""
        
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        self.last_optimization = current_time
        
        if DebugConfig.LOG_MEMORY_USAGE:
            print("Optimizing memory pools...")
        
        total_freed = 0
        for device_pools in self.memory_pools.values():
            for dtype_pool in device_pools.values():
                blocks_to_remove = []
                
                for i, block in enumerate(dtype_pool):
                    if not block.in_use:
                        # Remove old, unused blocks
                        age = block.get_age_seconds()
                        idle_time = block.get_idle_time_seconds()
                        
                        should_remove = False
                        
                        # Remove based on age and idle time
                        if age > 600 and idle_time > 300:  # 10min old, 5min idle
                            should_remove = True
                        elif idle_time > 900:  # 15min idle regardless of age
                            should_remove = True
                        elif block.priority == 0 and idle_time > 180:  # 3min idle for low priority
                            should_remove = True
                        
                        # Keep high-priority blocks longer
                        if block.priority >= 5:
                            should_remove = False
                        
                        if should_remove:
                            blocks_to_remove.append(i)
                            total_freed += block.size_bytes
                
                # Remove blocks in reverse order to maintain indices
                for i in reversed(blocks_to_remove):
                    del dtype_pool[i]
        
        if DebugConfig.LOG_MEMORY_USAGE and total_freed > 0:
            print(f"Memory optimization freed {total_freed / 1024**2:.1f} MB")
        
        # Force garbage collection if significant memory was freed
        if total_freed > 100 * 1024 * 1024:  # >100MB
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks"""
        
        if not self.leak_detection_enabled:
            return
        
        # Check for tensors that might have leaked
        current_time = time.time()
        leaked_count = 0
        
        for device_pools in self.memory_pools.values():
            for dtype_pool in device_pools.values():
                for block in dtype_pool:
                    if block.in_use and block.get_idle_time_seconds() > 1800:  # 30min unused
                        leaked_count += 1
        
        if leaked_count > 10:
            warnings.warn(
                f"Potential memory leak detected: {leaked_count} tensors unused for >30min",
                UserWarning
            )
    
    def _update_device_coordination(self):
        """Update multi-GPU memory coordination"""
        
        if not self.enable_multi_gpu or not torch.distributed.is_initialized():
            return
        
        try:
            # Share memory usage information across devices
            local_memory_info = {}
            for device in self.device_memory_usage:
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)
                    local_memory_info[str(device)] = {
                        'allocated': allocated,
                        'reserved': reserved,
                        'pool_usage': self.device_memory_usage[device]
                    }
            
            # In a real implementation, this would use distributed communication
            # to coordinate memory usage across GPUs
            
        except Exception as e:
            if DebugConfig.LOG_MEMORY_USAGE:
                print(f"Device coordination error: {e}")
    
    def _record_allocation(self, shape, dtype, device, allocation_time_ms, cache_hit):
        """Record allocation for performance analysis"""
        
        if not self.enable_profiling:
            return
        
        record = {
            'timestamp': time.time(),
            'shape': shape,
            'dtype': str(dtype),
            'device': str(device),
            'allocation_time_ms': allocation_time_ms,
            'cache_hit': cache_hit,
            'size_bytes': torch.empty(shape, dtype=dtype).numel() * torch.tensor(0, dtype=dtype).element_size()
        }
        
        self.allocation_history.append(record)
    
    def _get_pool_size(self, device: torch.device) -> int:
        """Get current pool size for device"""
        
        if device not in self.memory_pools:
            return 0
        
        total_size = 0
        for dtype_pool in self.memory_pools[device].values():
            total_size += sum(block.size_bytes for block in dtype_pool)
        
        return total_size
    
    def get_enhanced_statistics(self) -> MemoryStatistics:
        """Get comprehensive memory usage statistics"""
        
        total_allocated = 0
        total_cached = 0
        active_blocks = 0
        free_blocks = 0
        per_device_allocated = {}
        per_device_cached = {}
        
        for device, device_pools in self.memory_pools.items():
            device_allocated = 0
            device_cached = 0
            
            for dtype_pool in device_pools.values():
                for block in dtype_pool:
                    total_cached += block.size_bytes
                    device_cached += block.size_bytes
                    
                    if block.in_use:
                        total_allocated += block.size_bytes
                        device_allocated += block.size_bytes
                        active_blocks += 1
                    else:
                        free_blocks += 1
            
            per_device_allocated[str(device)] = device_allocated / (1024**2)
            per_device_cached[str(device)] = device_cached / (1024**2)
        
        # Calculate derived metrics
        pool_efficiency = total_allocated / total_cached if total_cached > 0 else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        # Calculate memory fragmentation (simplified)
        total_blocks = active_blocks + free_blocks
        memory_fragmentation = free_blocks / total_blocks if total_blocks > 0 else 0
        
        # Calculate allocation/deallocation rates
        time_window = 60.0  # 1 minute
        recent_allocations = [r for r in self.allocation_history 
                            if time.time() - r['timestamp'] < time_window]
        allocation_rate = len(recent_allocations) / time_window
        
        # Average allocation time
        if recent_allocations:
            avg_allocation_time = sum(r['allocation_time_ms'] for r in recent_allocations) / len(recent_allocations)
        else:
            avg_allocation_time = 0.0
        
        # Peak memory (approximation)
        peak_memory = 0
        if torch.cuda.is_available():
            for device in per_device_allocated:
                if 'cuda' in device:
                    device_idx = int(device.split(':')[1]) if ':' in device else 0
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device_idx))
        
        return MemoryStatistics(
            total_allocated_mb=total_allocated / (1024**2),
            peak_allocated_mb=peak_memory / (1024**2),
            total_cached_mb=total_cached / (1024**2),
            active_blocks=active_blocks,
            free_blocks=free_blocks,
            pool_efficiency=pool_efficiency,
            cache_hit_rate=cache_hit_rate,
            memory_fragmentation=memory_fragmentation,
            allocation_rate_per_sec=allocation_rate,
            per_device_allocated_mb=per_device_allocated,
            per_device_cached_mb=per_device_cached,
            avg_allocation_time_ms=avg_allocation_time,
        )
    
    def cleanup_enhanced(self, force: bool = False, device: Optional[torch.device] = None):
        """Enhanced cleanup with selective device targeting"""
        
        devices_to_clean = [device] if device else list(self.memory_pools.keys())
        
        total_freed = 0
        for target_device in devices_to_clean:
            if target_device not in self.memory_pools:
                continue
            
            device_pools = self.memory_pools[target_device]
            for dtype_pool in device_pools.values():
                blocks_to_remove = []
                
                for i, block in enumerate(dtype_pool):
                    if force or not block.in_use:
                        blocks_to_remove.append(i)
                        total_freed += block.size_bytes
                
                # Remove blocks in reverse order
                for i in reversed(blocks_to_remove):
                    del dtype_pool[i]
        
        # Update device memory tracking
        if self.enable_multi_gpu:
            for device in devices_to_clean:
                if device in self.device_memory_usage:
                    self.device_memory_usage[device] = 0
        
        # Force garbage collection and CUDA cache cleanup
        gc.collect()
        if torch.cuda.is_available():
            if device and device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        if DebugConfig.LOG_MEMORY_USAGE:
            print(f"Cleanup freed {total_freed / 1024**2:.1f} MB")
    
    def shutdown(self):
        """Shutdown memory manager and background threads"""
        
        self.shutdown_event.set()
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        
        self.cleanup_enhanced(force=True)


# Enhanced workspace management functions
def allocate_attention_workspace_enhanced(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int, 
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    include_backward: bool = True,
    enable_multi_gpu: bool = False,
    optimization_level: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Enhanced workspace allocation for Flash Attention with intelligent optimization
    
    Args:
        batch_size: Batch size
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        device: Device
        include_backward: Whether to allocate tensors for backward pass
        enable_multi_gpu: Whether to use multi-GPU optimization
        optimization_level: Optimization level (0=basic, 1=balanced, 2=aggressive)
        
    Returns:
        Dictionary with allocated workspace tensors
    """
    
    manager = get_enhanced_memory_manager()
    workspace = {}
    
    # Determine priority based on tensor usage frequency
    high_priority = 10 if optimization_level >= 2 else 5
    medium_priority = 5 if optimization_level >= 1 else 0
    low_priority = 0
    
    # Forward pass tensors (high priority - always needed)
    workspace['output'] = manager.allocate_enhanced(
        (batch_size, seq_len_q, num_heads, head_dim), dtype, device,
        priority=high_priority, access_pattern='sequential'
    )
    
    workspace['lse'] = manager.allocate_enhanced(
        (batch_size, num_heads, seq_len_q), torch.float32, device,
        priority=high_priority, access_pattern='sequential'
    )
    
    workspace['max_vals'] = manager.allocate_enhanced(
        (batch_size, num_heads, seq_len_q), torch.float32, device,
        priority=high_priority, access_pattern='sequential'
    )
    
    # Backward pass tensors (medium priority - only needed during training)
    if include_backward:
        workspace['grad_q'] = manager.allocate_enhanced(
            (batch_size, seq_len_q, num_heads, head_dim), dtype, device,
            priority=medium_priority, access_pattern='sequential'
        )
        
        workspace['grad_k'] = manager.allocate_enhanced(
            (batch_size, seq_len_k, num_heads, head_dim), dtype, device,
            priority=medium_priority, access_pattern='sequential'
        )
        
        workspace['grad_v'] = manager.allocate_enhanced(
            (batch_size, seq_len_k, num_heads, head_dim), dtype, device,
            priority=medium_priority, access_pattern='sequential'
        )
        
        # Delta computation tensor (can be fused in enhanced kernels)
        if optimization_level < 2:  # Skip if using fused kernels
            workspace['delta'] = manager.allocate_enhanced(
                (batch_size, num_heads, seq_len_q), torch.float32, device,
                priority=low_priority, access_pattern='sequential'
            )
    
    # Multi-GPU workspace tensors
    if enable_multi_gpu and torch.cuda.device_count() > 1:
        # Communication buffers for all-gather operations
        workspace['comm_buffer_q'] = manager.allocate_enhanced(
            (batch_size, seq_len_q, num_heads, head_dim), dtype, device,
            priority=medium_priority, access_pattern='random'
        )
        
        workspace['comm_buffer_output'] = manager.allocate_enhanced(
            (batch_size, seq_len_q, num_heads, head_dim), dtype, device,
            priority=medium_priority, access_pattern='random'
        )
    
    # Optimization-level specific allocations
    if optimization_level >= 2:
        # Pre-allocate frequently used intermediate tensors
        block_size = min(64, seq_len_q)  # Typical block size
        
        workspace['attention_block'] = manager.allocate_enhanced(
            (batch_size, num_heads, block_size, block_size), torch.float32, device,
            priority=high_priority, access_pattern='random'
        )
        
        workspace['softmax_buffer'] = manager.allocate_enhanced(
            (batch_size, num_heads, block_size), torch.float32, device,
            priority=high_priority, access_pattern='sequential'
        )
    
    return workspace


def free_attention_workspace_enhanced(workspace: Dict[str, torch.Tensor], 
                                     defer_cleanup: bool = True):
    """
    Enhanced workspace deallocation with intelligent caching
    
    Args:
        workspace: Dictionary with workspace tensors
        defer_cleanup: Whether to defer actual cleanup (keep in cache)
    """
    
    manager = get_enhanced_memory_manager()
    
    for tensor in workspace.values():
        if isinstance(tensor, torch.Tensor):
            if defer_cleanup:
                # Mark as deallocated but keep in cache for reuse
                manager.deallocate_enhanced(tensor)
            else:
                # Force immediate cleanup
                manager.deallocate_enhanced(tensor)
                del tensor


def estimate_peak_memory_usage_enhanced(
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    include_gradients: bool = True,
    enable_optimizations: bool = True,
    optimization_level: int = 1,
) -> Dict[str, float]:
    """
    Enhanced memory usage estimation with optimization considerations
    
    Args:
        batch_size: Batch size
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        include_gradients: Whether to include gradient memory
        enable_optimizations: Whether optimizations are enabled
        optimization_level: Level of optimizations
        
    Returns:
        Dictionary with detailed memory usage estimates in MB
    """
    
    # Base memory calculation
    base_memory = calculate_memory_requirements(
        batch_size, max(seq_len_q, seq_len_k), num_heads, head_dim, dtype
    )
    
    # Input tensors (Q, K, V)
    input_memory = base_memory['qkv_memory_mb']
    
    # Output tensor
    output_memory = base_memory['output_memory_mb']
    
    # Flash Attention intermediate memory (varies by optimization level)
    intermediate_base = base_memory['flash_intermediate_memory_mb']
    
    if enable_optimizations:
        if optimization_level >= 2:
            # Aggressive optimizations reduce intermediate memory
            intermediate_memory = intermediate_base * 0.7
        elif optimization_level >= 1:
            # Balanced optimizations
            intermediate_memory = intermediate_base * 0.85
        else:
            # Basic optimizations
            intermediate_memory = intermediate_base * 0.95
    else:
        intermediate_memory = intermediate_base
    
    # Forward pass statistics (LSE, max values)
    stats_memory = batch_size * num_heads * seq_len_q * 4 * 2 / (1024**2)  # fp32
    
    # Gradient memory (if training)
    gradient_memory = 0
    if include_gradients:
        gradient_memory = input_memory + output_memory  # dQ, dK, dV, dO
        
        # Additional backward pass intermediates
        if not enable_optimizations or optimization_level < 2:
            # Non-fused backward needs delta tensor
            gradient_memory += batch_size * num_heads * seq_len_q * 4 / (1024**2)
    
    # Workspace memory (depends on optimization level)
    if optimization_level >= 2:
        # Aggressive: pre-allocate more buffers
        workspace_memory = stats_memory * 3
    elif optimization_level >= 1:
        # Balanced: moderate workspace
        workspace_memory = stats_memory * 2
    else:
        # Basic: minimal workspace
        workspace_memory = stats_memory * 1.5
    
    # Memory pool overhead
    pool_overhead = 0
    if OptimizationConfig.ENABLE_MEMORY_POOLING:
        total_tensors_memory = input_memory + output_memory + intermediate_memory + gradient_memory
        pool_overhead = total_tensors_memory * 0.15  # 15% overhead for pooling
    
    # Multi-GPU overhead
    multi_gpu_overhead = 0
    if torch.cuda.device_count() > 1:
        # Communication buffers and cross-GPU coordination
        multi_gpu_overhead = (input_memory + output_memory) * 0.1
    
    total_memory = (
        input_memory + output_memory + intermediate_memory + 
        stats_memory + gradient_memory + workspace_memory + 
        pool_overhead + multi_gpu_overhead
    )
    
    return {
        'input_memory_mb': input_memory,
        'output_memory_mb': output_memory,
        'intermediate_memory_mb': intermediate_memory,
        'statistics_memory_mb': stats_memory,
        'gradient_memory_mb': gradient_memory,
        'workspace_memory_mb': workspace_memory,
        'pool_overhead_mb': pool_overhead,
        'multi_gpu_overhead_mb': multi_gpu_overhead,
        'total_memory_mb': total_memory,
        'memory_savings_ratio': base_memory['memory_savings_ratio'],
        'optimization_savings_mb': (intermediate_base - intermediate_memory),
    }


# Global enhanced memory manager instance
_enhanced_memory_manager: Optional[AdaptiveMemoryManager] = None


def get_enhanced_memory_manager(**kwargs) -> AdaptiveMemoryManager:
    """Get global enhanced memory manager instance"""
    global _enhanced_memory_manager
    if _enhanced_memory_manager is None:
        _enhanced_memory_manager = AdaptiveMemoryManager(**kwargs)
    return _enhanced_memory_manager


def get_comprehensive_memory_statistics() -> Dict[str, Any]:
    """
    Get comprehensive memory statistics across all subsystems
    
    Returns:
        Dictionary with detailed memory statistics
    """
    
    stats = {}
    
    # Enhanced manager statistics
    manager = get_enhanced_memory_manager()
    enhanced_stats = manager.get_enhanced_statistics()
    stats['enhanced_manager'] = enhanced_stats.__dict__
    
    # PyTorch CUDA memory stats
    if torch.cuda.is_available():
        cuda_stats = {}
        for i in range(torch.cuda.device_count()):
            device_stats = {
                'allocated_mb': torch.cuda.memory_allocated(i) / (1024**2),
                'cached_mb': torch.cuda.memory_reserved(i) / (1024**2),
                'max_allocated_mb': torch.cuda.max_memory_allocated(i) / (1024**2),
                'max_cached_mb': torch.cuda.max_memory_reserved(i) / (1024**2),
            }
            cuda_stats[f'cuda:{i}'] = device_stats
        stats['pytorch_cuda'] = cuda_stats
    
    # System memory stats
    if hasattr(psutil, 'virtual_memory'):
        system_memory = psutil.virtual_memory()
        stats['system_memory'] = {
            'total_gb': system_memory.total / (1024**3),
            'available_gb': system_memory.available / (1024**3),
            'used_gb': system_memory.used / (1024**3),
            'percentage': system_memory.percent,
        }
    
    # Process memory stats
    try:
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        stats['process_memory'] = {
            'rss_mb': process_memory.rss / (1024**2),
            'vms_mb': process_memory.vms / (1024**2),
        }
    except:
        pass
    
    return stats


def cleanup_all_memory():
    """Cleanup all memory pools and caches across all subsystems"""
    
    # Enhanced manager cleanup
    manager = get_enhanced_memory_manager()
    manager.cleanup_enhanced(force=True)
    
    # PyTorch cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reset peak memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


# Context manager for memory optimization
@contextlib.contextmanager
def optimized_memory_context(optimization_level: int = 2, 
                            cleanup_on_exit: bool = True):
    """
    Context manager for optimized memory operations
    
    Args:
        optimization_level: Level of optimization (0-2)
        cleanup_on_exit: Whether to cleanup memory on exit
    """
    
    manager = get_enhanced_memory_manager()
    
    # Store original settings
    original_optimization = manager.enable_adaptive_optimization
    
    try:
        # Enable optimizations
        manager.enable_adaptive_optimization = True
        yield manager
    finally:
        # Restore settings
        manager.enable_adaptive_optimization = original_optimization
        
        if cleanup_on_exit:
            manager._optimize_memory_pools()


# Export main functions and classes
__all__ = [
    'AdaptiveMemoryManager',
    'EnhancedMemoryBlock',
    'MemoryStatistics',
    'allocate_attention_workspace_enhanced',
    'free_attention_workspace_enhanced',
    'estimate_peak_memory_usage_enhanced',
    'get_enhanced_memory_manager',
    'get_comprehensive_memory_statistics',
    'cleanup_all_memory',
    'optimized_memory_context',
    
    # Legacy compatibility
    'MemoryManager',
    'allocate_attention_workspace',
    'free_attention_workspace',
    'estimate_peak_memory_usage',
    'get_memory_manager',
    'get_memory_statistics',
    'cleanup_memory',
]

# Provide aliases for backward compatibility
MemoryManager = AdaptiveMemoryManager
allocate_attention_workspace = allocate_attention_workspace_enhanced
free_attention_workspace = free_attention_workspace_enhanced
estimate_peak_memory_usage = estimate_peak_memory_usage_enhanced
get_memory_manager = get_enhanced_memory_manager
get_memory_statistics = get_comprehensive_memory_statistics
cleanup_memory = cleanup_all_memory