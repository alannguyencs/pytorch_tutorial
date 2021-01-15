import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange

#where checkpoints are temporarily saved
# = path_to_pytorch_tutorial + ckpt
from pathlib import Path
CWF = Path(__file__)
CKPT_PATH = str(CWF.parent.parent.parent.parent) + '/ckpt/'

# Hyper-parameters
NUM_CLASSES = 10
NUM_EPOCHES = 16
BATCH_SIZE = 64

MAX_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
LR_DECLINE_RATE = 0.88
LR_STEP = 32

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('device: ', device)