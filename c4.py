import torch
import numpy as np
from tqdm import trange
from utils import *

def newBoard(shape, device="cpu"):
    if len(shape) == 2: b, h, w = 1, *shape
    elif len(shape) == 3: b, h, w = shape
    else: assert 0, "invalid board shape"
    return torch.zeros((b, 2, h, w), dtype=torch.float32, device=torch.device(device), requires_grad=False)

def legalActions(boards:torch.Tensor):
    return torch.abs(1-torch.sum((boards.unsqueeze(0) if boards.ndim==3 else boards)[:,:,0], dim=1)).squeeze()

def getmask(cnct, device="cpu"):
    mask = np.zeros((cnct*2 + 2, 1, cnct, cnct))
    i = 0
    for c in range(cnct):
        h_ = np.ones((1,cnct))
        v_ = np.ones((cnct,1))
        
        hm = np.pad(h_, ((c, cnct-c-1), (0, 0)), constant_values=False)
        vm = np.pad(v_, ((0, 0), (c, cnct-c-1)), constant_values=False)
        mask[i] = hm
        mask[i+1] = vm
        i += 2
    mask[i] += np.eye(cnct)
    mask[i+1] += np.eye(cnct)[::-1]
    return torch.tensor(mask, dtype=torch.float32).to(device)

class valuator: 
        def __init__(self, cnct=4):
            self.m = getmask(cnct, device="cpu")
            self.mcuda = getmask(cnct, device="cuda")

        @torch.no_grad()
        def __call__(self, board):
            m = self.mcuda if board.is_cuda else self.m
            if board.ndim == 3: board = board.unsqueeze(0)
            zz0 = F.conv2d(board[:,0].unsqueeze(1), m)
            zz1 = F.conv2d(board[:,1].unsqueeze(1), m)
            v1 = 1*(torch.amax(zz0, dim=(1,2,3)) >= 4)
            v2 = 1*(torch.amax(zz1, dim=(1,2,3)) >= 4)
            return v1 - v2
value = valuator()#singleton class for determining winning states. We do this so we can use the same mask for all boards instead of passing it around during training


def buildBoard(actions, boardShape=(6, 7)):
    assert len(actions) == boardShape[1]
    b = newBoard(boardShape)
    for col in range(boardShape[1]):
        for a in actions[col]:
            b = drop(b, col, a)
    return b

@torch.no_grad()
def drop(boards:torch.Tensor, columns:torch.Tensor, color:int):
    newboards = (boards.unsqueeze(0) if boards.ndim==3 else boards).clone()
    batchsize, _, height, width = boards.shape
    iii = height - torch.sum(newboards, dim=(1,2), dtype=torch.int32).squeeze(dim=1) - 1
    bidxs = torch.arange(batchsize)
    newboards[bidxs,color,iii[bidxs,columns],columns] = 1
    return newboards

if __name__ == "__main__":

    b = newBoard((100,6,7))
    acts = torch.tensor(np.random.randint(0, 6, size=(100)))
    print(acts)


    b.to("cuda")
    for i in trange(1_000_000, ncols=120):
        b = drop(b, acts, 0)

