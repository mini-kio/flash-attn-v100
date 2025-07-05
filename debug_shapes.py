import torch

# Test tensor shapes for flash attention
batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)
v = torch.randn(batch_size, seq_len, num_heads, head_dim)

print(f"Original q shape: {q.shape}")
print(f"Original k shape: {k.shape}")
print(f"Original v shape: {v.shape}")

# Transpose to [batch, num_heads, seq_len, head_dim]
q_t = q.transpose(1, 2)
k_t = k.transpose(1, 2)
v_t = v.transpose(1, 2)

print(f"Transposed q shape: {q_t.shape}")
print(f"Transposed k shape: {k_t.shape}")
print(f"Transposed v shape: {v_t.shape}")

# Test matmul with corrected shapes
attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1))
print(f"attn_weights shape: {attn_weights.shape}")

# Test mask creation
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(f"initial mask shape: {mask.shape}")

mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
print(f"mask_expanded shape: {mask_expanded.shape}")

print(f"Can we apply mask? {attn_weights.shape == mask_expanded.shape}")

if attn_weights.shape == mask_expanded.shape:
    print("✅ Shapes match! The fix is correct.")
else:
    print("❌ Shapes still don't match.")
    print(f"Expected: {mask_expanded.shape}, Got: {attn_weights.shape}")

# Test bias shapes
bias_2d = torch.randn(seq_len, seq_len)
bias_3d = torch.randn(batch_size, seq_len, seq_len)

print(f"\nbias_2d shape: {bias_2d.shape}")
bias_2d_expanded = bias_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
print(f"bias_2d_expanded shape: {bias_2d_expanded.shape}")
print(f"Can add bias_2d? {attn_weights.shape == bias_2d_expanded.shape}")

print(f"\nbias_3d shape: {bias_3d.shape}")
bias_3d_expanded = bias_3d.unsqueeze(1).expand(-1, num_heads, -1, -1)
print(f"bias_3d_expanded shape: {bias_3d_expanded.shape}")
print(f"Can add bias_3d? {attn_weights.shape == bias_3d_expanded.shape}")

# Test output
output = torch.matmul(attn_weights, v_t)
print(f"\nOutput shape (before transpose): {output.shape}")
output_final = output.transpose(1, 2)
print(f"Final output shape: {output_final.shape}")
print(f"Final shape matches input? {output_final.shape == q.shape}")
