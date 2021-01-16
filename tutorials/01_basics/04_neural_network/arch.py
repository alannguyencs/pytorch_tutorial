from constants import *


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() #see [1] ReLU = max(0, x)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



"""
[1]: Why non-linear activation function?
A linear hidden layer is more of less useless because the composition of two linear functions is itself
a linear function. So unless you throw a non-linear in there, then you're not computing more than one
linear function even as you go deeper in the network.
Ref: https://www.coursera.org/lecture/neural-networks-deep-learning/why-do-you-need-non-linear-activation-functions-OASKH
"""