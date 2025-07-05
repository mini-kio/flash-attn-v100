# attention.py
"""
Main Flash Attention operations

This module provides the PyTorch autograd integration for Flash Attention V100.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from ..kernels.forward_kernel import (
    flash_attention_forward_triton,
    flash_attention_forward_autotuned,
)
from ..kernels.backward_kernel import flash_attention_backward_triton
from ..config import get_default_config, DebugConfig
from ..utils import validate_inputs, check_tensor_nan_inf

class FlashAttentionV100Function(torch.autograd.Function):
    """
    PyTorch autograd function for Flash Attention V100
    
    This class integrates the Triton kernels with PyTorch's automatic
    differentiation system, allowing for seamless use in neural networks.
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        scale: float,
        causal: bool = False,
        block_size_m: Optional[int] = None,
        block_size_n: Optional[int] = None,
        enable_autotuning: bool = True,
        return_softmax_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of Flash Attention
        
        Args:
            ctx: PyTorch autograd context for saving tensors
            q: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_k, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len_k, num_heads, head_dim]
            scale: Attention scale factor
            causal: Whether to apply causal masking
            block_size_m: Block size for M dimension
            block_size_n: Block size for N dimension
            enable_autotuning: Whether to use autotuning
            return_softmax_lse: Whether to return log-sum-exp values
            
        Returns:
            Output tensor, optionally with LSE values
        """
        
        # Validate inputs
        q, k, v, scale = validate_inputs(q, k, v, scale)
        
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_k, _, _ = k.shape
        
        # Get block sizes if not provided
        if block_size_m is None or block_size_n is None:
            config = get_default_config(max(seq_len_q, seq_len_k), head_dim)
            if block_size_m is None:
                block_size_m = config['BLOCK_M']
            if block_size_n is None:
                block_size_n = config['BLOCK_N']
        
        # Debug checks
        if DebugConfig.VALIDATE_OUTPUTS:
            check_tensor_nan_inf(q, "query")
            check_tensor_nan_inf(k, "key")
            check_tensor_nan_inf(v, "value")
        
        # Choose kernel based on autotuning preference
        if enable_autotuning:
            output, lse, max_vals = flash_attention_forward_autotuned(
                q, k, v, scale, causal
            )
        else:
            output, lse, max_vals = flash_attention_forward_triton(
                q, k, v, scale, causal, block_size_m, block_size_n
            )
        
        # Debug checks on output
        if DebugConfig.VALIDATE_OUTPUTS:
            check_tensor_nan_inf(output, "output")
            check_tensor_nan_inf(lse, "lse")
            check_tensor_nan_inf(max_vals, "max_vals")
        
        # Save tensors for backward pass
        if q.requires_grad or k.requires_grad or v.requires_grad:
            ctx.save_for_backward(q, k, v, output, lse, max_vals)
            ctx.scale = scale
            ctx.causal = causal
            ctx.block_size_m = block_size_m
            ctx.block_size_n = block_size_n
        
        if return_softmax_lse:
            return output, lse
        else:
            return output
    
    @staticmethod
    def backward(
        ctx, 
        grad_output: torch.Tensor,
        grad_softmax_lse: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass of Flash Attention
        
        Args:
            ctx: PyTorch autograd context with saved tensors
            grad_output: Gradient w.r.t. output
            grad_softmax_lse: Gradient w.r.t. LSE (if returned in forward)
            
        Returns:
            Gradients w.r.t. inputs (q, k, v, scale, ...)
        """
        
        # Retrieve saved tensors
        q, k, v, output, lse, max_vals = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        block_size_m = ctx.block_size_m
        block_size_n = ctx.block_size_n
        
        # Debug checks
        if DebugConfig.VALIDATE_OUTPUTS:
            check_tensor_nan_inf(grad_output, "grad_output")
        
        # Compute gradients using backward kernel
        grad_q, grad_k, grad_v = flash_attention_backward_triton(
            q, k, v, output, grad_output, lse, max_vals,
            scale, causal, block_size_m, block_size_n
        )
        
        # Debug checks on gradients
        if DebugConfig.VALIDATE_OUTPUTS:
            if grad_q is not None:
                check_tensor_nan_inf(grad_q, "grad_q")
            if grad_k is not None:
                check_tensor_nan_inf(grad_k, "grad_k")
            if grad_v is not None:
                check_tensor_nan_inf(grad_v, "grad_v")
        
        # Return gradients (None for non-tensor arguments)
        return (
            grad_q if q.requires_grad else None,
            grad_k if k.requires_grad else None, 
            grad_v if v.requires_grad else None,
            None,  # scale
            None,  # causal
            None,  # block_size_m
            None,  # block_size_n
            None,  # enable_autotuning
            None,  # return_softmax_lse
        )


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Standalone forward function for Flash Attention V100
    
    This function can be used when gradients are not needed.
    
    Args:
        q: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len_k, num_heads, head_dim]
        v: Value tensor [batch_size, seq_len_k, num_heads, head_dim]
        scale: Attention scale factor
        causal: Whether to apply causal masking
        **kwargs: Additional arguments passed to the kernel
        
    Returns:
        Output tensor [batch_size, seq_len_q, num_heads, head_dim]
    """
    
    # Validate inputs
    q, k, v, scale = validate_inputs(q, k, v, scale)
    
    # Get configuration
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    config = get_default_config(max(seq_len_q, seq_len_k), head_dim)
    
    # Extract block sizes from kwargs or use defaults
    block_size_m = kwargs.get('block_size_m', config['BLOCK_M'])
    block_size_n = kwargs.get('block_size_n', config['BLOCK_N'])
    enable_autotuning = kwargs.get('enable_autotuning', True)
    
    # Forward pass without gradients
    with torch.no_grad():
        if enable_autotuning:
            output, _, _ = flash_attention_forward_autotuned(q, k, v, scale, causal)
        else:
            output, _, _ = flash_attention_forward_triton(
                q, k, v, scale, causal, block_size_m, block_size_n
            )
    
    return output


def flash_attention_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    max_vals: torch.Tensor,
    scale: float,
    causal: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standalone backward function for Flash Attention V100
    
    Args:
        grad_output: Gradient w.r.t. output
        q: Query tensor from forward pass
        k: Key tensor from forward pass
        v: Value tensor from forward pass
        output: Output tensor from forward pass
        lse: Log-sum-exp values from forward pass
        max_vals: Max values from forward pass
        scale: Attention scale factor
        causal: Whether causal masking was used
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (grad_q, grad_k, grad_v)
    """
    
    # Get configuration
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    config = get_default_config(max(seq_len_q, seq_len_k), head_dim)
    
    # Extract block sizes
    block_size_m = kwargs.get('block_size_m', config['BLOCK_M'])
    block_size_n = kwargs.get('block_size_n', config['BLOCK_N'])
    
    # Backward pass
    grad_q, grad_k, grad_v = flash_attention_backward_triton(
        q, k, v, output, grad_output, lse, max_vals,
        scale, causal, block_size_m, block_size_n
    )
    
    return grad_q, grad_k, grad_v


def compare_with_pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """
    Compare Flash Attention V100 output with PyTorch's scaled_dot_product_attention
    
    This function is useful for validation and debugging.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        causal: Whether to apply causal masking
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    
    # Validate inputs
    q, k, v, scale = validate_inputs(q, k, v, None)
    
    # Flash Attention output
    flash_output = flash_attention_forward(q, k, v, scale, causal)
    
    # PyTorch reference output
    with torch.no_grad():
        # Convert to PyTorch's expected format: [batch, heads, seq, head_dim]
        q_pt = q.transpose(1, 2).contiguous()
        k_pt = k.transpose(1, 2).contiguous()
        v_pt = v.transpose(1, 2).contiguous()
        
        # Use PyTorch's implementation
        ref_output = F.scaled_dot_product_attention(
            q_pt, k_pt, v_pt,
            is_causal=causal,
            scale=scale
        )
        
        # Convert back to our format
        ref_output = ref_output.transpose(1, 2).contiguous()
    
    # Compare outputs
    max_diff = torch.max(torch.abs(flash_output - ref_output)).item()
    mean_diff = torch.mean(torch.abs(flash_output - ref_output)).item()
    allclose = torch.allclose(flash_output, ref_output, atol=atol, rtol=rtol)
    
    return {
        'max_absolute_difference': max_diff,
        'mean_absolute_difference': mean_diff,
        'allclose': allclose,
        'atol': atol,
        'rtol': rtol,
        'flash_output_shape': flash_output.shape,
        'reference_output_shape': ref_output.shape,
    }


# Export main functions
__all__ = [
    'FlashAttentionV100Function',
    'flash_attention_forward',
    'flash_attention_backward', 
    'compare_with_pytorch_attention',
]