import torch
import torch.nn.functional as F
import numpy as np
import torchviz
import torch.autograd.profiler as profiler
import re, os, time, numpy as np
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
    policies, valnets = [], []
    names = os.listdir(dir)
    names = sorted([int(re.search("[0-9]+", nam).group()) for nam in names])
    for name in names:
        policies.append(torch.load(f"{dir}\\v{name}.pth"))
        valnets.append(torch.load(f"{dir}\\p{name}.pth"))
    return policies, valnets

def brepr(board_, colors=True):
    board = board_.squeeze()
    assert board.ndim == 3, f"got board shape {board.shape}, expected a 3d tensor of shape (2, H, W)"
    d, h, w = board.shape
    if colors: string = gray
    else: string = ""
    for i in range(w):
        string += f" {i} "
    for y in range(h):
        string += "\n"
        for x in range(w):
            if board[0][y][x] == 1: string += (green + " O ") if colors else " O "
            elif board[1][y][x] == 1: string += (red + " X ") if colors else " X "
            else: string += (gray if colors else "") + " . "
    if colors: string += endc
    return string

def printBoard_(board_, colors=True):
    print(brepr(board_, colors=colors))

def printBoard(board_, colors=True):
    boards = board_.squeeze()
    if boards.ndim == 3: return printBoard_(boards, colors=colors)
    return [printBoard_(b, colors=colors) for b in boards]

if __name__ == "__main__":
    val = -1
    numTurns = 10
    discount = 0.9
    valueScale = 1
    #for i in trange(1_000_000, ncols=100):
    weights = rtg(val, numTurns, discount, valueScale)
    print(weights)
    print()

    val = torch.tensor([0, 1, -1, 1, 1, -1, 1])
    numTurns = torch.tensor([10, 5, 15, 23, 6, 4, 30])
    discount = 0.9
    valueScale = 1
    #for i in trange(1_000_000, ncols=100):
    weights = rtg(val, numTurns, discount, valueScale)
    print(weights)