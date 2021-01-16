from constants import *
from arch import ConvNet
from model import Model
from data import train_loader, test_loader


arch = ConvNet(NUM_CLASSES)
model = Model(arch)
model.train(train_loader)
model.test(test_loader)

