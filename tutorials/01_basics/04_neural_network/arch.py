from constants import *


# Fully connected neural network with one hidden layer
"""
1. Input layer
2. Neural network layers: fc1 + fc2
3. Output layer

In case no non-linear activation functions:
I
H1 = I x W1
O = H1 x W2
---> O = I x W1 x W2 = I x (W1 x W2) = I x W
---> equivalent to only 1 linear hidden layer
"""
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() #see [1] ReLU(x) = max(0, x)
        self.fc_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc_1(x)
        out = self.relu(out)
        out = self.fc_2(out)
        return out



"""
[1]: Why non-linear activation function?
A linear hidden layer is more of less useless because the composition of two linear functions is itself
a linear function. So unless you throw a non-linear in there, then you're not computing more than one
linear function even as you go deeper in the network.
Ref: https://www.coursera.org/lecture/neural-networks-deep-learning/why-do-you-need-non-linear-activation-functions-OASKH
"""