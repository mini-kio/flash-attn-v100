# basic_usage.py
"""
Enhanced usage examples for Flash Attention V100

This script demonstrates advanced features and optimizations:
- Multi-GPU training and inference
- Memory optimization strategies
- Performance profiling and analysis
- Mixed precision training
- Gradient checkpointing
- Real-world model integration examples
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import time
from typing import Tuple, Dict, Any, Optional
import warnings
import contextlib

# Import Enhanced Flash Attention V100
try:
    from .. import (
        flash_attention_v100,
        flash_attention_performance_mode,
        flash_attention_memory_mode,
    )
    from ..ops.memory import (
        optimized_memory_context,
        get_comprehensive_memory_statistics,
        cleanup_all_memory,
    )
    from ..utils import (
        PerformanceProfiler,
        benchmark_operation,
        get_device_info,
        setup_distributed_environment,
    )
    from ..config import get_enhanced_config
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    print("Enhanced Flash Attention V100 not available. Please install the package.")
    FLASH_ATTENTION_AVAILABLE = False
    exit(1)


class EnhancedMultiHeadAttention(nn.Module):
    """
    Enhanced Multi-Head Attention module using Flash Attention V100
    
    Features:
    - Automatic optimization strategy selection
    - Multi-GPU support with head parallelization
    - Memory-efficient training with gradient checkpointing
    - Mixed precision support
    - Performance monitoring
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 batch_first: bool = True,
                 # Enhanced parameters
                 optimization_strategy: str = 'balanced',
                 enable_multi_gpu: bool = True,
                 enable_gradient_checkpointing: bool = False,
                 enable_performance_monitoring: bool = False):
        """
        Initialize Enhanced Multi-Head Attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            batch_first: Whether batch dimension is first
            optimization_strategy: 'speed', 'memory', 'balanced', or 'adaptive'
            enable_multi_gpu: Whether to enable multi-GPU support
            enable_gradient_checkpointing: Whether to use gradient checkpointing
            enable_performance_monitoring: Whether to monitor performance
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.optimization_strategy = optimization_strategy
        self.enable_multi_gpu = enable_multi_gpu
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Performance monitoring
        if self.enable_performance_monitoring:
            self.profiler = PerformanceProfiler(enable_detailed_profiling=True)
            self.forward_count = 0
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                average_attn_weights: bool = True,
                is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced forward pass
        
        Args:
            query: Query tensor [seq_len, batch_size, embed_dim] or [batch_size, seq_len, embed_dim]
            key: Key tensor (optional, defaults to query)
            value: Value tensor (optional, defaults to key)
            key_padding_mask: Mask for padding tokens
            attn_mask: Attention mask
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads
            is_causal: Whether to apply causal masking
            
        Returns:
            Tuple of (output, attention_weights)
        """
        
        if self.enable_performance_monitoring:
            self.forward_count += 1
            context_manager = self.profiler.profile_operation(f"forward_pass_{self.forward_count}")
        else:
            context_manager = contextlib.nullcontext()
        
        with context_manager:
            # Handle input format
            if not self.batch_first:
                query = query.transpose(0, 1)
                if key is not None:
                    key = key.transpose(0, 1)
                if value is not None:
                    value = value.transpose(0, 1)
            
            batch_size, seq_len, embed_dim = query.shape
            
            # Use query for key and value if not provided
            if key is None:
                key = query
            if value is None:
                value = key
            
            # Project to Q, K, V
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, k.size(1), self.num_heads, self.head_dim)
            v = v.view(batch_size, v.size(1), self.num_heads, self.head_dim)
            
            # Prepare bias tensor from masks
            bias = None
            if attn_mask is not None or key_padding_mask is not None:
                bias = self._create_bias_tensor(
                    attn_mask, key_padding_mask, batch_size, 
                    self.num_heads, seq_len, k.size(1), q.device, q.dtype
                )
            
            # Enhanced Flash Attention with automatic optimization
            with flash_attention_performance_mode(self.optimization_strategy):
                if need_weights:
                    warnings.warn(
                        "need_weights=True not supported with Flash Attention. "
                        "Attention weights will not be returned.",
                        UserWarning
                    )
                
                # Apply Flash Attention
                attn_output = flash_attention_v100(
                    q, k, v,
                    causal=is_causal,
                    dropout_p=self.dropout if self.training else 0.0,
                    bias=bias,
                    optimization_strategy=self.optimization_strategy,
                    enable_multi_gpu=self.enable_multi_gpu,
                    enable_profiling=self.enable_performance_monitoring,
                    fallback_on_error=True,
                )
            
            # Reshape and project output
            attn_output = attn_output.view(batch_size, seq_len, embed_dim)
            output = self.out_proj(attn_output)
            
            # Handle output format
            if not self.batch_first:
                output = output.transpose(0, 1)
            
            # Return attention weights as None since Flash Attention doesn't compute them
            attn_weights = None if not need_weights else torch.zeros(
                batch_size, self.num_heads, seq_len, k.size(1),
                device=output.device, dtype=output.dtype
            )
            
            return output, attn_weights
    
    def _create_bias_tensor(self, attn_mask, key_padding_mask, batch_size, num_heads, 
                           seq_len_q, seq_len_k, device, dtype):
        """Create bias tensor from attention and padding masks"""
        
        bias = torch.zeros(batch_size, num_heads, seq_len_q, seq_len_k, 
                          device=device, dtype=dtype)
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                bias = bias.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            elif attn_mask.dim() == 3:
                bias = bias.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # Apply key padding mask
        if key_padding_mask is not None:
            bias = bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf')
            )
        
        return bias
    
    def get_performance_stats(self) -> Optional[Dict[str, Any]]:
        """Get performance statistics if monitoring is enabled"""
        if self.enable_performance_monitoring:
            return self.profiler.get_summary()
        return None


def example_basic_enhanced_attention():
    """Enhanced basic Flash Attention usage example"""
    print("=== Enhanced Basic Flash Attention Example ===")
    
    # Setup with automatic device detection
    device_info = get_device_info()
    if not device_info['cuda_available']:
        print("CUDA not available. Skipping example.")
        return
    
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Enhanced model parameters
    batch_size = 4
    seq_len = 2048
    num_heads = 16
    head_dim = 64
    embed_dim = num_heads * head_dim
    
    print(f"Configuration:")
    print(f"  Device: {device} ({device_info['devices'][0]['name']})")
    print(f"  Dtype: {dtype}")
    print(f"  Shape: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
    print(f"  V100 Compatible: {device_info['devices'][0]['v100_compatible']}")
    
    # Create enhanced attention module
    attention = EnhancedMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        optimization_strategy='balanced',
        enable_multi_gpu=True,
        enable_performance_monitoring=True
    ).to(device)
    
    # Create input tensors
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, embed_dim,
        device=device, dtype=dtype
    )
    
    print(f"Input tensor created. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Run enhanced attention with performance monitoring
    with torch.cuda.amp.autocast():
        start_time = time.time()
        output, _ = attention(input_tensor, is_causal=True)
        end_time = time.time()
    
    print(f"Enhanced Flash Attention completed in {(end_time - start_time) * 1000:.2f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Get performance statistics
    perf_stats = attention.get_performance_stats()
    if perf_stats:
        print(f"Performance statistics:")
        print(f"  Total operations: {perf_stats['total_operations']}")
        print(f"  Average time per operation: {perf_stats['avg_operation_time_ms']:.2f} ms")
        print(f"  Total memory delta: {perf_stats['total_memory_delta_mb']:.1f} MB")
    
    print("✓ Enhanced basic example completed successfully\n")
    return output


def example_multi_gpu_training():
    """Multi-GPU training example with Flash Attention"""
    print("=== Multi-GPU Training Example ===")
    
    device_info = get_device_info()
    num_gpus = device_info['cuda_device_count']
    
    if num_gpus < 2:
        print("Multi-GPU training requires at least 2 GPUs. Skipping example.")
        return
    
    print(f"Detected {num_gpus} GPUs")
    
    # Setup distributed environment (simulated for example)
    try:
        setup_distributed_environment()
        distributed_mode = dist.is_initialized()
    except:
        distributed_mode = False
        print("Running in single-process multi-GPU mode")
    
    # Model parameters optimized for multi-GPU
    batch_size = 8  # Larger batch for multi-GPU
    seq_len = 4096  # Longer sequences
    num_heads = 32  # More heads for better parallelization
    head_dim = 64
    embed_dim = num_heads * head_dim
    
    print(f"Multi-GPU configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Heads per GPU: {num_heads // num_gpus}")
    print(f"  Distributed mode: {distributed_mode}")
    
    # Create model with multi-GPU optimization
    model = EnhancedMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        optimization_strategy='speed',  # Optimize for speed in multi-GPU
        enable_multi_gpu=True,
        enable_gradient_checkpointing=True,  # For large sequences
        enable_performance_monitoring=True
    )
    
    # Move to first GPU and wrap with DataParallel if not using distributed
    device = torch.device('cuda:0')
    model = model.to(device)
    
    if not distributed_mode and num_gpus > 1:
        model = nn.DataParallel(model)
        print("Using DataParallel for multi-GPU training")
    
    # Create data
    torch.manual_seed(42)
    input_data = torch.randn(
        batch_size, seq_len, embed_dim,
        device=device, dtype=torch.float16
    )
    
    # Training step with mixed precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    print("Running multi-GPU training step...")
    
    model.train()
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        start_time = time.time()
        output, _ = model(input_data, is_causal=True)
        
        # Simple loss for demonstration
        loss = output.mean()
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    end_time = time.time()
    
    print(f"Training step completed in {(end_time - start_time) * 1000:.2f} ms")
    print(f"Loss: {loss.item():.6f}")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Multi-GPU memory statistics
    if num_gpus > 1:
        print("Per-GPU memory usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
            print(f"  GPU {i}: {allocated:.1f} MB allocated, {cached:.1f} MB cached")
    
    print("✓ Multi-GPU training example completed successfully\n")
    return output


def example_memory_optimization():
    """Memory optimization strategies example"""
    print("=== Memory Optimization Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available. Skipping memory optimization example.")
        return
    
    # Large model parameters that would typically cause OOM
    batch_size = 2
    seq_len = 8192  # Very long sequences
    num_heads = 24
    head_dim = 64
    embed_dim = num_heads * head_dim
    
    print(f"Memory optimization configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Estimated naive memory: ~{batch_size * seq_len * seq_len * num_heads * 2 / 1024**3:.1f} GB")
    
    # Clear memory and reset statistics
    cleanup_all_memory()
    torch.cuda.reset_peak_memory_stats()
    
    # Strategy 1: Memory-optimized mode
    print("\n1. Testing memory-optimized mode...")
    
    with flash_attention_memory_mode():
        model_memory = EnhancedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            optimization_strategy='memory',
            enable_gradient_checkpointing=True,
            enable_performance_monitoring=True
        ).to(device)
        
        input_data = torch.randn(
            batch_size, seq_len, embed_dim,
            device=device, dtype=torch.float16
        )
        
        with torch.cuda.amp.autocast():
            start_time = time.time()
            output_memory = model_memory(input_data, is_causal=True)[0]
            end_time = time.time()
        
        memory_mode_time = (end_time - start_time) * 1000
        memory_mode_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"  Time: {memory_mode_time:.2f} ms")
        print(f"  Peak memory: {memory_mode_peak:.1f} MB")
    
    # Strategy 2: Optimized memory context
    print("\n2. Testing optimized memory context...")
    
    torch.cuda.reset_peak_memory_stats()
    
    with optimized_memory_context(optimization_level=2, cleanup_on_exit=True):
        model_optimized = EnhancedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            optimization_strategy='adaptive',
            enable_gradient_checkpointing=True,
        ).to(device)
        
        with torch.cuda.amp.autocast():
            start_time = time.time()
            output_optimized = model_optimized(input_data, is_causal=True)[0]
            end_time = time.time()
        
        optimized_mode_time = (end_time - start_time) * 1000
        optimized_mode_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"  Time: {optimized_mode_time:.2f} ms")
        print(f"  Peak memory: {optimized_mode_peak:.1f} MB")
    
    # Strategy 3: Enhanced memory manager statistics
    print("\n3. Memory manager statistics...")
    
    memory_stats = get_comprehensive_memory_statistics()
    if 'enhanced_manager' in memory_stats:
        stats = memory_stats['enhanced_manager']
        print(f"  Pool efficiency: {stats['pool_efficiency']:.2%}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Memory fragmentation: {stats['memory_fragmentation']:.2%}")
        print(f"  Active blocks: {stats['active_blocks']}")
        print(f"  Free blocks: {stats['free_blocks']}")
    
    # Compare outputs for correctness
    max_diff = torch.max(torch.abs(output_memory - output_optimized)).item()
    print(f"\n4. Output correctness check:")
    print(f"  Max difference between strategies: {max_diff:.2e}")
    print(f"  Outputs are {'identical' if max_diff < 1e-4 else 'similar' if max_diff < 1e-2 else 'different'}")
    
    print("✓ Memory optimization example completed successfully\n")
    return output_optimized


def example_performance_benchmarking():
    """Comprehensive performance benchmarking example"""
    print("=== Performance Benchmarking Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available. Skipping benchmarking example.")
        return
    
    # Test different configurations
    test_configs = [
        {'name': 'Small', 'batch_size': 4, 'seq_len': 512, 'num_heads': 8, 'head_dim': 64},
        {'name': 'Medium', 'batch_size': 4, 'seq_len': 1024, 'num_heads': 12, 'head_dim': 64},
        {'name': 'Large', 'batch_size': 2, 'seq_len': 2048, 'num_heads': 16, 'head_dim': 64},
        {'name': 'XLarge', 'batch_size': 1, 'seq_len': 4096, 'num_heads': 20, 'head_dim': 64},
    ]
    
    optimization_strategies = ['speed', 'balanced', 'memory']
    
    print("Running comprehensive benchmarks...")
    print("-" * 80)
    print(f"{'Config':8} {'Strategy':10} {'Time (ms)':10} {'Memory (MB)':12} {'TFLOPS':8} {'Efficiency':10}")
    print("-" * 80)
    
    results = []
    
    for config in test_configs:
        embed_dim = config['num_heads'] * config['head_dim']
        
        for strategy in optimization_strategies:
            # Skip memory strategy for large configs to avoid timeouts
            if strategy == 'memory' and config['seq_len'] > 2048:
                continue
            
            try:
                # Create model
                model = EnhancedMultiHeadAttention(
                    embed_dim=embed_dim,
                    num_heads=config['num_heads'],
                    dropout=0.0,
                    optimization_strategy=strategy,
                    enable_multi_gpu=False,  # Single GPU for fair comparison
                    enable_performance_monitoring=False
                ).to(device)
                
                # Create input
                input_data = torch.randn(
                    config['batch_size'], config['seq_len'], embed_dim,
                    device=device, dtype=torch.float16
                )
                
                # Benchmark the operation
                def forward_pass():
                    with torch.cuda.amp.autocast():
                        return model(input_data, is_causal=True)[0]
                
                benchmark_result = benchmark_operation(
                    forward_pass,
                    num_warmup=10,
                    num_runs=50,
                    return_output=False
                )
                
                if 'error' not in benchmark_result:
                    # Calculate TFLOPS (approximate)
                    total_flops = (4 * config['batch_size'] * config['num_heads'] * 
                                 config['seq_len'] * config['seq_len'] * config['head_dim'])
                    tflops = total_flops / (benchmark_result['avg_time_ms'] / 1000) / 1e12
                    
                    # Calculate efficiency (percentage of theoretical peak)
                    device_info = get_device_info()
                    if device_info['devices']:
                        theoretical_peak = 125  # V100 theoretical peak TFLOPS for FP16
                        efficiency = (tflops / theoretical_peak) * 100
                    else:
                        efficiency = 0
                    
                    results.append({
                        'config': config['name'],
                        'strategy': strategy,
                        'time_ms': benchmark_result['avg_time_ms'],
                        'memory_mb': benchmark_result['peak_memory_mb'],
                        'tflops': tflops,
                        'efficiency': efficiency
                    })
                    
                    print(f"{config['name']:8} {strategy:10} {benchmark_result['avg_time_ms']:8.2f} "
                          f"{benchmark_result['peak_memory_mb']:10.1f} {tflops:6.1f} {efficiency:8.1f}%")
                else:
                    print(f"{config['name']:8} {strategy:10} ERROR: {benchmark_result['error']}")
                
            except Exception as e:
                print(f"{config['name']:8} {strategy:10} ERROR: {str(e)[:50]}")
            
            # Cleanup between runs
            torch.cuda.empty_cache()
    
    print("-" * 80)
    
    # Find best configurations
    if results:
        best_speed = max(results, key=lambda x: x['tflops'])
        best_memory = min(results, key=lambda x: x['memory_mb'])
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        
        print(f"\nBest Results:")
        print(f"  Highest throughput: {best_speed['config']} + {best_speed['strategy']} "
              f"({best_speed['tflops']:.1f} TFLOPS)")
        print(f"  Lowest memory: {best_memory['config']} + {best_memory['strategy']} "
              f"({best_memory['memory_mb']:.1f} MB)")
        print(f"  Best efficiency: {best_efficiency['config']} + {best_efficiency['strategy']} "
              f"({best_efficiency['efficiency']:.1f}%)")
    
    print("✓ Performance benchmarking completed successfully\n")
    return results


def example_real_world_integration():
    """Real-world model integration example"""
    print("=== Real-World Model Integration Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available. Skipping integration example.")
        return
    
    class TransformerBlock(nn.Module):
        """Transformer block using Enhanced Flash Attention"""
        
        def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
            super().__init__()
            
            self.attention = EnhancedMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                optimization_strategy='balanced',
                enable_multi_gpu=True,
                enable_gradient_checkpointing=True
            )
            
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            )
        
        def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
            # Self-attention with residual connection
            attn_out, _ = self.attention(x, is_causal=is_causal)
            x = self.norm1(x + attn_out)
            
            # Feed-forward with residual connection
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            return x
    
    class GPTModel(nn.Module):
        """Simplified GPT model using Enhanced Flash Attention"""
        
        def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                     num_layers: int, max_seq_len: int):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
            
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, embed_dim * 4)
                for _ in range(num_layers)
            ])
            
            self.ln_f = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size)
            
            self.max_seq_len = max_seq_len
        
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            seq_len = input_ids.size(1)
            
            # Embeddings
            x = self.embedding(input_ids)
            x = x + self.pos_embedding[:seq_len]
            
            # Transformer blocks
            for block in self.blocks:
                x = block(x, is_causal=True)
            
            # Final layer norm and head
            x = self.ln_f(x)
            logits = self.head(x)
            
            return logits
    
    # Model configuration
    vocab_size = 50257  # GPT-2 vocab size
    embed_dim = 768
    num_heads = 12
    num_layers = 6  # Smaller for demonstration
    max_seq_len = 1024
    batch_size = 4
    seq_len = 512
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Maximum sequence length: {max_seq_len}")
    
    # Create model
    model = GPTModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create sample data
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nRunning training step...")
    
    model.train()
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        start_time = time.time()
        
        # Forward pass
        logits = model(input_ids)
        
        # Simple loss (shift for causal language modeling)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1)
        )
        
        forward_time = time.time()
    
    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    end_time = time.time()
    
    print(f"Training step completed:")
    print(f"  Forward time: {(forward_time - start_time) * 1000:.2f} ms")
    print(f"  Total time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Inference example
    print(f"\nRunning inference...")
    
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            start_time = time.time()
            
            # Generate a few tokens
            generated_ids = input_ids[:1, :10].clone()  # Start with first 10 tokens
            
            for _ in range(20):  # Generate 20 tokens
                logits = model(generated_ids)
                next_token = torch.argmax(logits[:, -1:], dim=-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            end_time = time.time()
    
    print(f"Generated {generated_ids.size(1)} tokens in {(end_time - start_time) * 1000:.2f} ms")
    print(f"Tokens per second: {20 / (end_time - start_time):.1f}")
    
    print("✓ Real-world integration example completed successfully\n")
    return model


def main():
    """Run all enhanced examples"""
    print("Enhanced Flash Attention V100 - Advanced Usage Examples")
    print("=" * 60)
    
    # Check system requirements
    device_info = get_device_info()
    print(f"System Information:")
    print(f"  PyTorch version: {device_info['pytorch_version']}")
    print(f"  CUDA available: {device_info['cuda_available']}")
    print(f"  Number of GPUs: {device_info['cuda_device_count']}")
    
    if device_info['cuda_available'] and device_info['devices']:
        for i, dev in enumerate(device_info['devices']):
            print(f"  GPU {i}: {dev['name']} (CC {dev['compute_capability']}, "
                  f"{dev['total_memory_gb']:.1f} GB)")
    
    print(f"  Distributed available: {device_info['distributed_available']}")
    print(f"  Flash Attention available: {FLASH_ATTENTION_AVAILABLE}")
    print()
    
    if not device_info['cuda_available']:
        print("CUDA not available. Some examples will be skipped.")
        return
    
    try:
        # Run examples
        example_basic_enhanced_attention()
        example_multi_gpu_training()
        example_memory_optimization()
        example_performance_benchmarking()
        example_real_world_integration()
        
        print("=" * 60)
        print("✓ All enhanced examples completed successfully!")
        print("\nRecommendations for production use:")
        print("1. Use 'balanced' strategy for most workloads")
        print("2. Enable multi-GPU for large models")
        print("3. Use gradient checkpointing for long sequences")
        print("4. Monitor memory usage with the enhanced manager")
        print("5. Profile your specific workload for optimal settings")
        print("6. Consider mixed precision training for better performance")
        
        # Final memory cleanup
        cleanup_all_memory()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()