from constants import *
from arch import ResNet, ResidualBlock
from model import Model
from data import train_loader, test_loader


arch = ResNet(ResidualBlock, [2, 2, 2])
model = Model(arch)
model.train(train_loader)
model.test(test_loader)