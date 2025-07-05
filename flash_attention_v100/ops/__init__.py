"""
Operations package for Flash Attention V100

This package contains high-level operations and memory management
for Flash Attention implementation.
"""

from .attention import (
    FlashAttentionV100Function,
    flash_attention_forward,
    flash_attention_backward,
    compare_with_pytorch_attention,
)

from .memory import (
    MemoryManager,
    allocate_attention_workspace,
    free_attention_workspace,
    estimate_peak_memory_usage,
    get_memory_statistics,
)

# Version information
__version__ = "0.1.0"

# Re-export main operations
__all__ = [
    # Main attention operations
    'FlashAttentionV100Function',
    'flash_attention_forward',
    'flash_attention_backward',
    'compare_with_pytorch_attention',
    
    # Memory management
    'MemoryManager',
    'allocate_attention_workspace',
    'free_attention_workspace', 
    'estimate_peak_memory_usage',
    'get_memory_statistics',
]

# Configuration for operations
OPERATION_DEFAULTS = {
    'enable_memory_pool': True,
    'enable_workspace_reuse': True,
    'enable_gradient_checkpointing': False,
    'memory_efficiency_mode': 'balanced',  # 'speed', 'balanced', 'memory'
}

def get_operation_defaults():
    """Get default operation configuration"""
    return OPERATION_DEFAULTS.copy()

def set_memory_efficiency_mode(mode: str):
    """
    Set memory efficiency mode
    
    Args:
        mode: One of 'speed', 'balanced', 'memory'
            - 'speed': Optimize for speed, higher memory usage
            - 'balanced': Balance speed and memory usage  
            - 'memory': Optimize for memory, potentially slower
    """
    if mode not in ['speed', 'balanced', 'memory']:
        raise ValueError(f"Invalid mode {mode}. Must be one of: speed, balanced, memory")
    
    OPERATION_DEFAULTS['memory_efficiency_mode'] = mode
    
    # Adjust other settings based on mode
    if mode == 'speed':
        OPERATION_DEFAULTS['enable_memory_pool'] = True
        OPERATION_DEFAULTS['enable_workspace_reuse'] = True
        OPERATION_DEFAULTS['enable_gradient_checkpointing'] = False
    elif mode == 'balanced':
        OPERATION_DEFAULTS['enable_memory_pool'] = True
        OPERATION_DEFAULTS['enable_workspace_reuse'] = True
        OPERATION_DEFAULTS['enable_gradient_checkpointing'] = False
    elif mode == 'memory':
        OPERATION_DEFAULTS['enable_memory_pool'] = False
        OPERATION_DEFAULTS['enable_workspace_reuse'] = False
        OPERATION_DEFAULTS['enable_gradient_checkpointing'] = True

# Global memory manager instance
_global_memory_manager = None

def get_global_memory_manager():
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def cleanup_global_memory():
    """Cleanup global memory manager"""
    global _global_memory_manager
    if _global_memory_manager is not None:
        _global_memory_manager.cleanup()
        _global_memory_manager = None