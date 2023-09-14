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
    device = val.device if torch.is_tensor(val) else "cpu"
    exps = torch.arange(numTurns, 0, -1, device=device)
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

def printValueAttr(values, memPositions, numSteps=-1):
    cols = [purple, blue, cyan, lime, lemon, red, pink, orange, green, gray]
    rep = ""
    c = 0
    for i, val in enumerate(values[:numSteps]):
        for j, g in enumerate(memPositions):
            if i in g: rep += cols[j]; break
        rep += f" {val.item()}, "
        c += 1
        if c >= sum([1 for e in memPositions if e[-1] >= i]): rep += "\n"; c = 0
        rep += endc
    print(rep + endc)



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