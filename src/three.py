import torch 

# Data vs Parameter using Autograd
# Data is a tensor that is not trainable, while a parameter is a tensor that is trainable. In PyTorch, parameters are typically used to represent the weights and biases of a neural network, while data tensors are used to represent input data or intermediate results. 

# A standard data tensor 
x_data = torch.tensor([[1., 2.], [3., 4.]])
print(f"Data Tensor requires_grad: {x_data.requires_grad}")

# A parameter tensor
x_param = torch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Parameter Tensor requires_grad: {x_param.requires_grad}")

# Outputs:
# Data Tensor requires_grad: False
# Parameter Tensor requires_grad: True

