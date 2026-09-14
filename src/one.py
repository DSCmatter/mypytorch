# Direct creationg from data 
from random import Random

import torch 

# Input - a standard python list 
data = [[1,2,3], [4,5,6]]
my_tensor = torch.tensor(data)

print(my_tensor)

# output: a tensor object
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Pattern 2: Creation from a Desired Shape 
# we know the shape you need, but not the values yet. This is how you initialize a model weights.

shape = (2,3)
# Create a tensor of zeros with the desired shape

ones = torch.ones(shape)
zeroes = torch.zeros(shape)
random = torch.randn(shape)

print(f"Random Tensor: \n{random}")

# Output: a tensor object
# Random Tensor: 
# tensor([[ 0.9498,  1.8853, -1.2557],
#         [ 0.2863, -1.3753,  0.1714]])

# Pattern 3: Creatiion from mimicking an existing tensor
# You need a new tensor with the same shape as an existing tensor, but with different values. 

# Input: A template tensor 
template_tensor = torch.tensor([[1,2], [4,5]])

# Create a new tensor with the same shape as the template tensor, but filled with ones
rand_like = torch.randn_like(template_tensor, dtype=torch.float)

print(f"Template Tensor: \n{template_tensor}")
print(f"Random Tensor like template: \n{rand_like}")

# Outputs: 
# Template Tensor: 
# tensor([[1, 2],
#         [4, 5]])
# Random Tensor like template: 
# tensor([[0.1415, 0.8079],
#         [0.9899, 1.5586]])
