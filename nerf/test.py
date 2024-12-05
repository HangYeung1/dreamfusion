import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

rawData = np.load("tiny_nerf_data.npz",allow_pickle=True)
images = rawData["images"]
poses = rawData["poses"]
focal = rawData["focal"]

print(poses.shape)