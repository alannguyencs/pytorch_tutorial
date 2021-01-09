import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#where checkpoints are temporarily saved
# = path_to_pytorch_tutorial + ckpt
from pathlib import Path
CWF = Path(__file__)
CKPT_PATH = str(CWF.parent.parent.parent.parent) + '/ckpt/'

# Hyper-parameters
INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_EPOCHES = 60
LEARNING_RATE = 1e-3