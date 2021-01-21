from constants import *

# Image preprocessing modules
augmentation_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='../../data',
                               train=True,
                               transform=augmentation_transform,
                               download=True)

test_dataset = datasets.CIFAR10(root='../../data',
                              train=False,
                              transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

print ("Length train loader = {}, test loader = {}".format(len(train_loader), len(test_loader)))