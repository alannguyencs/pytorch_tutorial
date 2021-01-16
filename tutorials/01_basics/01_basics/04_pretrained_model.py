import torch
import torchvision
import torch.nn as nn



# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)


# Freeze resnet parameters
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
# requires_grad = True by default
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

"""
Case A: param.requires_grad = False   || Case B: param.requires_grad = True
In case A, params have learnt to differentiate several features, are adaptive with various categories.
In case B, params would changed to only work on 100 categories in training data.

LR = 0.01
Case A: pretrained params do not change. fc params change significantly.
Case B: all params change significantly.
Best strategy: change pretrained params with small lr (e.g. 1e-5), set lr = big (e.g. 1e-2) to fc params.
"""
