# test_attention.py
"""
Test suite for Flash Attention V100

This module contains comprehensive tests to ensure correctness and performance
of the Flash Attention implementation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

try:
    from flash_attention_v100 import (
        flash_attention_v100,
        flash_attention_forward,
        compare_with_pytorch_attention,
        FlashAttentionV100Function,
    )
    from flash_attention_v100.config import SUPPORTED_DTYPES
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    pytestmark = pytest.mark.skip("Flash Attention V100 not available")

class TestFlashAttentionV100:
    """Test class for Flash Attention V100"""
    
    @pytest.fixture
    def device(self):
        """Test device fixture"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device('cuda')
    
    @pytest.fixture(params=[
        (2, 512, 8, 64),
        (1, 1024, 12, 64),
        (4, 256, 16, 32),
        (2, 2048, 8, 128),
    ])
    def tensor_shapes(self, request):
        """Test tensor shapes fixture"""
        return request.param  # (batch_size, seq_len, num_heads, head_dim)
    
    @pytest.fixture(params=SUPPORTED_DTYPES)
    def dtype(self, request):
        """Test data types fixture"""
        return request.param
    
    def create_test_tensors(
        self, 
        batch_size: int, 
        seq_len: int, 
        num_heads: int, 
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test tensors with specified properties"""
        
        # Use a fixed seed for reproducibility
        torch.manual_seed(42)
        
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
        
        # Normalize to prevent overflow in attention computation
        q = q / (head_dim ** 0.5)
        k = k / (head_dim ** 0.5)
        
        return q, k, v
    
    def test_basic_forward(self, tensor_shapes, dtype, device):
        """Test basic forward pass functionality"""
        batch_size, seq_len, num_heads, head_dim = tensor_shapes
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        # Test forward pass
        output = flash_attention_v100(q, k, v)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, num_heads, head_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        # Check output device and dtype
        assert output.device == device, f"Output device mismatch: expected {device}, got {output.device}"
        assert output.dtype == dtype, f"Output dtype mismatch: expected {dtype}, got {output.dtype}"
    
    def test_causal_attention(self, tensor_shapes, dtype, device):
        """Test causal attention functionality"""
        batch_size, seq_len, num_heads, head_dim = tensor_shapes
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        # Test causal attention
        output_causal = flash_attention_v100(q, k, v, causal=True)
        output_non_causal = flash_attention_v100(q, k, v, causal=False)
        
        # Outputs should be different (unless very special case)
        assert not torch.allclose(output_causal, output_non_causal, atol=1e-6), \
            "Causal and non-causal outputs are too similar"
        
        # Check shapes
        assert output_causal.shape == output_non_causal.shape
    
    def test_backward_pass(self, tensor_shapes, dtype, device):
        """Test backward pass and gradient computation"""
        batch_size, seq_len, num_heads, head_dim = tensor_shapes
        
        # Skip bfloat16 for gradient tests due to precision issues
        if dtype == torch.bfloat16:
            pytest.skip("bfloat16 not suitable for gradient precision tests")
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device, requires_grad=True
        )
        
        # Forward pass
        output = flash_attention_v100(q, k, v)
        
        # Create a simple loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        assert q.grad is not None, "Query gradient is None"
        assert k.grad is not None, "Key gradient is None"
        assert v.grad is not None, "Value gradient is None"
        
        assert torch.isfinite(q.grad).all(), "Query gradient contains non-finite values"
        assert torch.isfinite(k.grad).all(), "Key gradient contains non-finite values"
        assert torch.isfinite(v.grad).all(), "Value gradient contains non-finite values"
        
        # Check gradient shapes
        assert q.grad.shape == q.shape, "Query gradient shape mismatch"
        assert k.grad.shape == k.shape, "Key gradient shape mismatch"
        assert v.grad.shape == v.shape, "Value gradient shape mismatch"
    
    def test_numerical_accuracy(self, device):
        """Test numerical accuracy against PyTorch reference"""
        # Use smaller tensors for precise comparison
        batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 64
        dtype = torch.float32  # Use fp32 for best precision
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        # Compare with PyTorch implementation
        comparison = compare_with_pytorch_attention(q, k, v, causal=False)
        
        # Check accuracy
        assert comparison['allclose'], f"Outputs not close enough: max_diff={comparison['max_absolute_difference']}"
        assert comparison['max_absolute_difference'] < 1e-4, "Maximum difference too large"
    
    def test_causal_numerical_accuracy(self, device):
        """Test causal attention numerical accuracy"""
        batch_size, seq_len, num_heads, head_dim = 2, 128, 4, 64
        dtype = torch.float32
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        # Compare causal attention
        comparison = compare_with_pytorch_attention(q, k, v, causal=True)
        
        assert comparison['allclose'], f"Causal outputs not close enough: max_diff={comparison['max_absolute_difference']}"
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency for large sequences"""
        # Test with larger sequence length
        batch_size, seq_len, num_heads, head_dim = 1, 4096, 8, 64
        dtype = torch.float16
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        tensors_memory = torch.cuda.memory_allocated() - initial_memory
        
        # Run Flash Attention
        try:
            output = flash_attention_v100(q, k, v)
            peak_memory = torch.cuda.max_memory_allocated() - initial_memory
            
            # Memory should be much less than naive O(n²) implementation
            # Naive attention would need batch_size * num_heads * seq_len * seq_len * 2 bytes
            naive_attention_memory = batch_size * num_heads * seq_len * seq_len * 2
            
            print(f"Tensors memory: {tensors_memory / 1024**2:.1f} MB")
            print(f"Peak memory: {peak_memory / 1024**2:.1f} MB")
            print(f"Naive attention would need: {naive_attention_memory / 1024**2:.1f} MB")
            
            # Flash Attention should use significantly less memory
            assert peak_memory < naive_attention_memory * 0.1, \
                "Memory usage not significantly reduced compared to naive attention"
                
        except torch.cuda.OutOfMemoryError:
            pytest.fail("Out of memory error - Flash Attention should be more memory efficient")
        finally:
            torch.cuda.empty_cache()
    
    def test_different_block_sizes(self, device):
        """Test different block size configurations"""
        batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
        dtype = torch.float16
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device
        )
        
        block_sizes = [(32, 32), (64, 64), (128, 64)]
        outputs = []
        
        for block_m, block_n in block_sizes:
            output = flash_attention_v100(
                q, k, v, 
                block_size_m=block_m,
                block_size_n=block_n,
                enable_autotuning=False
            )
            outputs.append(output)
        
        # All outputs should be very similar
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-4, rtol=1e-4), \
                f"Outputs with different block sizes differ too much"
    
    def test_gradient_correctness(self, device):
        """Test gradient correctness using finite differences"""
        batch_size, seq_len, num_heads, head_dim = 1, 64, 2, 32
        dtype = torch.float64  # Use double precision for accurate finite differences
        
        q, k, v = self.create_test_tensors(
            batch_size, seq_len, num_heads, head_dim, dtype, device, requires_grad=True
        )
        
        # Define a simple loss function
        def loss_fn(q, k, v):
            output = flash_attention_v100(q, k, v)
            return torch.sum(output**2)
        
        # Use torch's gradcheck
        test_passed = torch.autograd.gradcheck(
            lambda q, k, v: loss_fn(q, k, v),
            (q, k, v),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        
        assert test_passed, "Gradient check failed"
    
    @pytest.mark.slow
    def test_performance_benchmark(self, device):
        """Basic performance benchmark"""
        from flash_attention_v100 import benchmark_flash_attention
        
        results = benchmark_flash_attention(
            batch_size=2,
            seq_len=1024,
            num_heads=12,
            head_dim=64,
            dtype=torch.float16,
            num_warmup=5,
            num_runs=20
        )
        
        print(f"Performance results: {results}")
        
        # Basic sanity checks
        assert results['avg_time_ms'] > 0, "Invalid timing result"
        assert results['throughput_tflops'] > 0, "Invalid throughput result"
        assert results['memory_savings_ratio'] > 1, "No memory savings detected"
    
    def test_edge_cases(self, device):
        """Test edge cases and error conditions"""
        # Test with very small tensors
        q = torch.randn(1, 1, 1, 16, device=device, dtype=torch.float16)
        k = torch.randn(1, 1, 1, 16, device=device, dtype=torch.float16)
        v = torch.randn(1, 1, 1, 16, device=device, dtype=torch.float16)
        
        output = flash_attention_v100(q, k, v)
        assert output.shape == q.shape
        
        # Test mismatched shapes (should raise error)
        q_wrong = torch.randn(1, 2, 1, 16, device=device, dtype=torch.float16)
        
        with pytest.raises(ValueError):
            flash_attention_v100(q_wrong, k, v)
    
    def test_mixed_precision(self, device):
        """Test mixed precision scenarios"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        batch_size, seq_len, num_heads, head_dim = 2, 256, 8, 64
        
        # Test with different input/output precisions
        with torch.cuda.amp.autocast():
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            
            output = flash_attention_v100(q, k, v)
            
            # Output should be in reduced precision
            assert output.dtype in [torch.float16, torch.bfloat16]


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "-s"])