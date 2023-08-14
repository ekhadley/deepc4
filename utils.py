import torch
import numpy as np
import torchviz
import torch.autograd.profiler as profiler
import os, time, numpy as np
from tqdm import trange


purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
lime = '\033[92m'
yellow = '\033[93m'
red = "\033[38;5;196m"
pink = "\033[38;5;206m"
orange = "\033[38;5;202m"
green = "\033[38;5;34m"
gray = "\033[38;5;8m"

bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def printBoard(board):
    str = gray
    for i in range(board.shape[1]):
        str += f" {i} "
    str += "\n"
    for row in board:
        str += "\n"
        for col in row:
            if col == 1: str += green + " O "
            elif col == -1: str += red + " X "
            else: str += gray + " . "
    str += endc
    print(str)

def sampleOpponents(numOpponents, weight=2):
    probs = [(i+1)**weight for i in range(numOpponents)]
    dist = torch.distributions.Categorical(torch.tensor(probs))
    return dist.sample().item()

def rtg(val, numTurns, discount, valueScale):
    weights = [valueScale*val*discount**(numTurns-i-1) for i in range(numTurns)]
    return weights

def loadAllModels(dir):
    models = []
    names = os.listdir(dir)
    names.sort(key=lambda x:int(x.replace(".pth", "")) )
    for name in names:
        models.append(torch.load(os.path.join(dir, name)))
    return models

def boardFromObs(obs):
    state = np.float32(obs)
    board = state[0] - state[1]
    return board