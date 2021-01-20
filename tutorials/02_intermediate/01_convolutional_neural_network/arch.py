from constants import *


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.name = 'CNN'
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


"""
Batch normalisation has learnable parameters, because it includes an affine transformation.
Since the norm is calculated per channel, the parameters γ and β are vectors of size num_channels 
(one element per channel), which results in an individual scale and shift per channel. 
As with any other learnable parameter in PyTorch, they need to be created with a fixed size, 
hence you need to specify the number of channels
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
"""