import torch
import torch.nn.functional as F
import numpy as np
import torchviz
import torch.autograd.profiler as profiler
import os, time, numpy as np
from tqdm import trange


purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
lime = '\033[92m'
lemon = '\033[93m'
red = "\033[38;5;196m"
pink = "\033[38;5;206m"
orange = "\033[38;5;202m"
green = "\033[38;5;34m"
gray = "\033[38;5;8m"

bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def sampleOpponents(numOpponents, weight=2):
    probs = [(i+1)**weight for i in range(numOpponents)]
    dist = torch.distributions.Categorical(torch.tensor(probs))
    return dist.sample().item()

def rtg(val, numTurns, discount, valueScale=1):
    exps = torch.arange(numTurns, 0, -1, device=val.device)
    discs = torch.pow(discount, exps)
    return val*discs*valueScale

def loadAllModels(dir):
    models = []
    names = os.listdir(dir)
    names.sort(key=lambda x:int(x.replace(".pth", "")) )
    for name in names:
        models.append(torch.load(os.path.join(dir, name)))
    return models

def printBoard_(board_):
    board = board_.squeeze()
    assert board.ndim == 3, f"got board shape {board.shape}, expected a 3d tensor of shape (2, H, W)"
    d, h, w = board.shape
    str = gray
    for i in range(w):
        str += f" {i} "
    for y in range(h):
        str += "\n"
        for x in range(w):
            if board[0][y][x] == 1: str += green + " O "
            elif board[1][y][x] == 1: str += red + " X "
            else: str += gray + " . "
    str += endc
    print(str)

def printBoard(board_):
    boards = board_.squeeze()
    if boards.ndim == 3: return printBoard_(boards)
    return [printBoard_(b) for b in boards]



if __name__ == "__main__":
    val = -1
    numTurns = 10
    discount = 0.9
    valueScale = 1
    weights = rtg(val, numTurns, discount, valueScale)
    print(weights)

    for i in trange(100_000):
        exps = torch.arange(numTurns, -1, -1)
        discs = val*discount**(exps)
    print(discs)