import torch
import numpy as np
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


def sampleOpponents(numOpponents, weight=2):
    probs = [(i+1)**weight for i in range(numOpponents)]
    dist = torch.distributions.Categorical(torch.tensor(probs))
    return dist.sample().item()

def rtg(val, numTurns, discount, valueScale, endScale=None):
    endScale = 1 if endScale is None else endScale
    weights = [valueScale*val*discount**(numTurns-i-1) for i in range(numTurns)]
    weights[-1] *= endScale
    return weights