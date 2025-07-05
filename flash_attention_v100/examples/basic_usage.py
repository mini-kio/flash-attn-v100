# basic_usage.py
"""
Basic usage examples for Flash Attention V100

This script demonstrates how to use Flash Attention V100 in various scenarios.
"""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any

# Import Flash Attention V100
try:
    from flash_attention_v100 import (
        flash_attention_v100,
        benchmark_flash_attention,
        compare_with_pytorch_attention,
    )
    from flash_attention_v100.ops.memory import get_memory_statistics, cleanup_memory
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    print("Flash Attention V100 not available. Please install the package.")
    FLASH_ATTENTION_AVAILABLE = False
    exit(1)


def example_basic_attention():
    """Basic Flash Attention usage example"""
    print("=== Basic Flash Attention Example ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    # Model parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 12
    head_dim = 64
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Shape: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
    
    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    print(f"Input tensors created. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Run Flash Attention
    start_time = time.time()
    output = flash_attention_v100(q, k, v)
    end_time = time.time()
    
    print(f"Flash Attention completed in {(end_time - start_time) * 1000:.2f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Check for NaN/Inf
    if torch.isnan(output).any():
        print("WARNING: Output contains NaN values!")
    if torch.isinf(output).any():
        print("WARNING: Output contains Inf values!")
    
    print("✓ Basic example completed successfully\n")
    return output


def example_causal_attention():
    """Causal attention example (for autoregressive models)"""
    print("=== Causal Attention Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    batch_size = 1
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create input tensors
    torch.manual_seed(123)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Compare causal vs non-causal attention
    output_causal = flash_attention_v100(q, k, v, causal=True)
    output_non_causal = flash_attention_v100(q, k, v, causal=False)
    
    # Verify causal property: output should be different
    diff = torch.abs(output_causal - output_non_causal).mean()
    print(f"Mean difference between causal and non-causal: {diff:.6f}")
    
    # Verify causal masking worked
    assert diff > 1e-4, "Causal and non-causal outputs are too similar!"
    
    print("✓ Causal attention working correctly\n")
    return output_causal


def example_gradient_computation():
    """Example with gradient computation (for training)"""
    print("=== Gradient Computation Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32  # Use fp32 for better gradient precision
    
    batch_size = 2
    seq_len = 256
    num_heads = 8
    head_dim = 64
    
    # Create input tensors with gradients enabled
    torch.manual_seed(456)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Forward pass
    output = flash_attention_v100(q, k, v)
    
    # Create a simple loss (sum of squares)
    loss = (output ** 2).sum()
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Gradients computed:")
    print(f"  Q gradient shape: {q.grad.shape}, norm: {q.grad.norm().item():.6f}")
    print(f"  K gradient shape: {k.grad.shape}, norm: {k.grad.norm().item():.6f}")
    print(f"  V gradient shape: {v.grad.shape}, norm: {v.grad.norm().item():.6f}")
    
    # Verify gradients are finite
    assert torch.isfinite(q.grad).all(), "Q gradient contains non-finite values"
    assert torch.isfinite(k.grad).all(), "K gradient contains non-finite values"
    assert torch.isfinite(v.grad).all(), "V gradient contains non-finite values"
    
    print("✓ Gradient computation working correctly\n")
    return output, loss


def example_memory_efficiency():
    """Demonstrate memory efficiency of Flash Attention"""
    print("=== Memory Efficiency Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    # Test with larger sequence length to show memory benefits
    batch_size = 1
    seq_len = 4096  # Large sequence length
    num_heads = 8
    head_dim = 64
    
    print(f"Testing with large sequence length: {seq_len}")
    
    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create input tensors
    torch.manual_seed(789)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    input_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Input memory: {input_memory:.1f} MB")
    
    # Run Flash Attention
    output = flash_attention_v100(q, k, v)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    
    # Calculate what naive attention would use
    naive_attention_memory = batch_size * num_heads * seq_len * seq_len * 2 / 1024**2  # fp16
    print(f"Naive attention would need: {naive_attention_memory:.1f} MB for attention matrix alone")
    
    memory_saved = naive_attention_memory / peak_memory
    print(f"Memory savings ratio: {memory_saved:.1f}x")
    
    print("✓ Memory efficiency demonstrated\n")
    return output


def example_performance_comparison():
    """Compare performance with PyTorch's attention"""
    print("=== Performance Comparison Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    batch_size = 2
    seq_len = 1024
    num_heads = 12
    head_dim = 64
    
    # Create input tensors
    torch.manual_seed(101112)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Compare numerical accuracy
    print("Checking numerical accuracy...")
    comparison = compare_with_pytorch_attention(q, k, v, causal=False, atol=1e-3, rtol=1e-3)
    
    print(f"Numerical comparison results:")
    print(f"  Max absolute difference: {comparison['max_absolute_difference']:.6f}")
    print(f"  Mean absolute difference: {comparison['mean_absolute_difference']:.6f}")
    print(f"  Outputs are close: {comparison['allclose']}")
    
    if not comparison['allclose']:
        print("WARNING: Outputs differ significantly from PyTorch reference!")
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    benchmark_results = benchmark_flash_attention(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_warmup=5,
        num_runs=20
    )
    
    print(f"Performance results:")
    print(f"  Average time: {benchmark_results['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {benchmark_results['throughput_tflops']:.2f} TFLOPS")
    print(f"  Memory usage: {benchmark_results['memory_usage_mb']:.1f} MB")
    print(f"  Memory savings: {benchmark_results['memory_savings_ratio']:.1f}x")
    
    print("✓ Performance comparison completed\n")
    return benchmark_results


def example_different_configurations():
    """Test different block size configurations"""
    print("=== Different Configurations Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create input tensors
    torch.manual_seed(131415)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Test different block size configurations
    block_configs = [(32, 32), (64, 64), (128, 64)]
    results = {}
    
    for block_m, block_n in block_configs:
        print(f"Testing block configuration: {block_m}x{block_n}")
        
        start_time = time.time()
        output = flash_attention_v100(
            q, k, v,
            block_size_m=block_m,
            block_size_n=block_n,
            enable_autotuning=False
        )
        end_time = time.time()
        
        results[(block_m, block_n)] = {
            'output': output,
            'time_ms': (end_time - start_time) * 1000,
        }
        
        print(f"  Time: {results[(block_m, block_n)]['time_ms']:.2f} ms")
    
    # Verify all configurations produce similar results
    reference_output = results[(64, 64)]['output']
    for config, result in results.items():
        if config != (64, 64):
            max_diff = torch.max(torch.abs(result['output'] - reference_output))
            print(f"Max difference from reference ({config}): {max_diff:.6f}")
            
            if max_diff > 1e-3:
                print(f"WARNING: Large difference detected for config {config}")
    
    print("✓ Different configurations tested\n")
    return results


def example_memory_monitoring():
    """Monitor memory usage during Flash Attention"""
    print("=== Memory Monitoring Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    batch_size = 1
    seq_len = 2048
    num_heads = 8
    head_dim = 64
    
    # Get initial memory statistics
    initial_stats = get_memory_statistics()
    print(f"Initial memory usage:")
    print(f"  CUDA allocated: {initial_stats['cuda_memory']['allocated_mb']:.1f} MB")
    print(f"  CUDA cached: {initial_stats['cuda_memory']['cached_mb']:.1f} MB")
    
    # Create tensors and run attention
    torch.manual_seed(161718)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Memory after tensor creation
    after_creation_stats = get_memory_statistics()
    print(f"\nAfter tensor creation:")
    print(f"  CUDA allocated: {after_creation_stats['cuda_memory']['allocated_mb']:.1f} MB")
    print(f"  Memory pool allocated: {after_creation_stats['memory_pool']['total_allocated_mb']:.1f} MB")
    
    # Run Flash Attention
    output = flash_attention_v100(q, k, v)
    
    # Memory after computation
    after_computation_stats = get_memory_statistics()
    print(f"\nAfter computation:")
    print(f"  CUDA allocated: {after_computation_stats['cuda_memory']['allocated_mb']:.1f} MB")
    print(f"  Peak CUDA allocated: {after_computation_stats['cuda_memory']['max_allocated_mb']:.1f} MB")
    print(f"  Pool efficiency: {after_computation_stats['memory_pool']['pool_efficiency']:.2f}")
    
    # Cleanup
    cleanup_memory()
    
    final_stats = get_memory_statistics()
    print(f"\nAfter cleanup:")
    print(f"  CUDA allocated: {final_stats['cuda_memory']['allocated_mb']:.1f} MB")
    
    print("✓ Memory monitoring completed\n")
    return output


def main():
    """Run all examples"""
    print("Flash Attention V100 - Basic Usage Examples")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Some examples may not work properly.")
        return
    
    # Check V100 compatibility
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        print("WARNING: GPU compute capability is below 7.0. Flash Attention V100 is optimized for V100+.")
    
    try:
        # Run examples
        example_basic_attention()
        example_causal_attention()
        example_gradient_computation()
        example_memory_efficiency()
        example_performance_comparison()
        example_different_configurations()
        example_memory_monitoring()
        
        print("=" * 50)
        print("✓ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try integrating Flash Attention V100 into your model")
        print("2. Experiment with different block sizes for your workload")
        print("3. Use autotuning for optimal performance")
        print("4. Monitor memory usage in your training loops")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_memory()


if __name__ == "__main__":
    main()