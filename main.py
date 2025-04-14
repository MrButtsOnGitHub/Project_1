import torch
from model import NeuralNet  # Importing the model

# Creating a tensor
x = torch.tensor([[1, 2], [3, 4]])
print(x)

# Creating a random tensor
y = torch.rand(2, 2)
print(y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

x = torch.randn(3, requires_grad=True)  # Enable gradient tracking
y = x * 2
z = y.mean()
z.backward()  # Compute gradients
print(x.grad)  # Gradient of z with respect to x

# Load and print the model
model = NeuralNet()
print(model)
