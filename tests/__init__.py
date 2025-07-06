# __init__.py
"""
Test suite for Flash Attention V100

This package contains comprehensive tests for Flash Attention V100 implementation.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for importing flash_attention_v100
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_CONFIG = {
    'default_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'default_dtype': torch.float16,
    'tolerance_fp16': 1e-3,
    'tolerance_fp32': 1e-5,
    'tolerance_bfloat16': 1e-2,
    'skip_slow_tests': False,
    'benchmark_iterations': 10,
}

# Test fixtures and utilities
def get_test_device():
    """Get device for testing"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')

def check_v100_available():
    """Check if V100 or compatible GPU is available"""
    if not torch.cuda.is_available():
        return False
    
    capability = torch.cuda.get_device_capability()
    return capability >= (7, 0)  # V100 is compute capability 7.0

def skip_if_no_v100():
    """Skip test if V100 is not available"""
    if not check_v100_available():
        pytest.skip("V100 or compatible GPU not available")

# Shared device fixture
@pytest.fixture
def device():
    """Return CUDA device, skipping or xfail when appropriate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device('cuda')
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        pytest.xfail(reason="GPU compute capability < 7.0, optimizations may not be optimal")
    return dev

def create_test_tensors(
    batch_size: int,
    seq_len: int, 
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
    requires_grad: bool = False,
    seed: int = 42
):
    """
    Create standardized test tensors
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype: Data type
        device: Device
        requires_grad: Whether tensors require gradients
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (q, k, v) tensors
    """
    if device is None:
        device = get_test_device()
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        dtype=dtype, device=device, requires_grad=requires_grad
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        dtype=dtype, device=device, requires_grad=requires_grad
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        dtype=dtype, device=device, requires_grad=requires_grad
    )
    
    # Normalize to prevent overflow
    scale = (head_dim ** -0.5)
    q = q * scale
    k = k * scale
    
    return q, k, v

def get_tolerance_for_dtype(dtype: torch.dtype) -> float:
    """Get appropriate numerical tolerance for data type"""
    if dtype == torch.float32:
        return TEST_CONFIG['tolerance_fp32']
    elif dtype == torch.float16:
        return TEST_CONFIG['tolerance_fp16']
    elif dtype == torch.bfloat16:
        return TEST_CONFIG['tolerance_bfloat16']
    else:
        return TEST_CONFIG['tolerance_fp32']

# Test categories
class TestCategories:
    """Test category markers"""
    
    # Test speed markers
    FAST = pytest.mark.fast
    SLOW = pytest.mark.slow
    
    # Test type markers
    UNIT = pytest.mark.unit
    INTEGRATION = pytest.mark.integration
    BENCHMARK = pytest.mark.benchmark
    
    # Hardware requirement markers
    V100_REQUIRED = pytest.mark.v100_required
    MULTI_GPU = pytest.mark.multi_gpu
    
    # Precision markers
    FP16 = pytest.mark.fp16
    FP32 = pytest.mark.fp32
    BFLOAT16 = pytest.mark.bfloat16

# Common test configurations
COMMON_SHAPES = [
    (1, 64, 8, 64),    # Small batch, small sequence
    (2, 256, 12, 64),  # Medium batch, medium sequence
    (1, 1024, 8, 64),  # Large sequence
    (4, 128, 16, 32),  # Large batch, small head_dim
    (2, 512, 8, 128),  # Large head_dim
]

DTYPES_TO_TEST = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

BLOCK_SIZE_CONFIGS = [
    (32, 32),
    (64, 64),
    (128, 64),
    (64, 128),
]

# Test utilities
def assert_allclose_with_dtype(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = None,
    atol: float = None,
    msg: str = ""
):
    """Assert tensors are close with dtype-appropriate tolerance"""
    
    if rtol is None:
        rtol = get_tolerance_for_dtype(actual.dtype)
    if atol is None:
        atol = get_tolerance_for_dtype(actual.dtype)
    
    try:
        torch.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        max_diff = torch.max(torch.abs(actual - expected))
        mean_diff = torch.mean(torch.abs(actual - expected))
        raise AssertionError(
            f"{msg}\nMax difference: {max_diff}\nMean difference: {mean_diff}\nOriginal error: {e}"
        )

def run_memory_test(test_func, *args, **kwargs):
    """
    Run test with memory monitoring
    
    Args:
        test_func: Test function to run
        *args, **kwargs: Arguments for test function
        
    Returns:
        Tuple of (result, peak_memory_mb)
    """
    if not torch.cuda.is_available():
        return test_func(*args, **kwargs), 0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    result = test_func(*args, **kwargs)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    return result, peak_memory

# Export test utilities
__all__ = [
    'TEST_CONFIG',
    'get_test_device',
    'check_v100_available',
    'skip_if_no_v100',
    'device',
    'create_test_tensors',
    'get_tolerance_for_dtype',
    'TestCategories',
    'COMMON_SHAPES',
    'DTYPES_TO_TEST',
    'BLOCK_SIZE_CONFIGS',
    'assert_allclose_with_dtype',
    'run_memory_test',
]

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "fast: fast running tests")
    config.addinivalue_line("markers", "slow: slow running tests") 
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "benchmark: benchmark tests")
    config.addinivalue_line("markers", "v100_required: requires V100 GPU")
    config.addinivalue_line("markers", "multi_gpu: requires multiple GPUs")
    config.addinivalue_line("markers", "fp16: tests for float16 precision")
    config.addinivalue_line("markers", "fp32: tests for float32 precision")
    config.addinivalue_line("markers", "bfloat16: tests for bfloat16 precision")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration"""
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip V100 tests if not available
    if not check_v100_available():
        skip_v100 = pytest.mark.skip(reason="V100 GPU not available")
        for item in items:
            if "v100_required" in item.keywords:
                item.add_marker(skip_v100)