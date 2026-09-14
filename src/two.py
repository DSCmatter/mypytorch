import torch 

tensor = torch.randn(2,3)

print(f"Shape: {tensor.shape}")
print(f"Data Type: {tensor.dtype}")
print(f"Device: {tensor.device}")

# Outputs: 
# Shape: torch.Size([2, 3])
# Data Type: torch.float32
# Device: cpu