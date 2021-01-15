from constants import *
from arch import RNN
from model import Model
from data import train_loader, test_loader


arch = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
model = Model(arch)
model.train(train_loader)
model.test(test_loader)