from constants import *


# MNIST dataset (images and labels)
train_dataset = datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          shuffle=False)

