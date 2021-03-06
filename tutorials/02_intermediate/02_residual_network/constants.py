import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import os

#where checkpoints are temporarily saved
# = path_to_pytorch_tutorial + ckpt
from pathlib import Path
CWF = Path(__file__)
CKPT_PATH = str(CWF.parent.parent.parent.parent) + '/ckpt/'

# Hyper-parameters
NUM_CLASSES = 10
NUM_EPOCHS = 2
BATCH_SIZE = 64

MAX_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
LR_DECLINE_RATE = 0.88
LR_STEP = 32

INF = 1e18

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ('device: ', device)