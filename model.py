import torch.nn as nn

# Define a simple model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Fully connected layer

    def forward(self, x):
        return self.fc1(x)