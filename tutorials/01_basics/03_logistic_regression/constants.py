import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#where checkpoints are temporarily saved
# = path_to_pytorch_tutorial + ckpt
from pathlib import Path
CWF = Path(__file__)
CKPT_PATH = str(CWF.parent.parent.parent.parent) + '/ckpt/'

# Hyper-parameters
MODEL_NAME = 'logistic_regression_model'
INPUT_SIZE = 784
NUM_CLASSES = 10
NUM_EPOCHES = 16
BATCH_SIZE = 64
LEARNING_RATE = 1e-3