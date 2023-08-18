import torch
import numpy as np
import os, time, numpy as np
from tqdm import trange
from utils import *


def newBoard(shape, device="cpu"):
    if len(shape) == 2: b = 1; h, w = shape
    elif len(shape) == 3: b, h, w = shape
    return torch.zeros((b, 2, h, w), dtype=torch.float32, device=torch.device(device), requires_grad=False)

def legalActions(boards_:torch.tensor):
    boards = boards_.clone()
    if boards.dim() == 3: boards = boards_.unsqueeze(0)
    boards = torch.sum(boards,axis=1)
    return 1*(boards[:,0,:]==0).squeeze()

def drop(boards_:torch.tensor, columns:torch.tensor, color:int):
    boards = boards_.clone()
    batchsize, _, height, width = boards.shape
    b = torch.sum(boards, axis=1).squeeze(axis=1)
    arange = torch.arange(batchsize)
    cols = b[arange,:,columns]
    occs = [torch.where(cols[i]==0)[0][-1] for i in range(batchsize)]
    boards[arange, color, occs, columns] = 1
    return boards

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
            zz0 = F.conv2d(board[:,0], m)
            zz1 = F.conv2d(board[:,1], m)
            v1 = 1*(torch.amax(zz0, dim=(0,1,2))/4 >= 1)
            v2 = -1*(torch.amax(zz1, dim=(0,1,2))/4 >= 1)
            return v1 + v2
value = valuator()#singleton class for determining winning states. We do this so we can use the same mask for all boards instead of passing it around during training

def buildBoard(actions, boardShape=(6, 7)):
    assert len(actions) == boardShape[1]
    b = newBoard(boardShape)
    for col in range(boardShape[1]):
        for a in actions[col]:
            b = drop(b, col, a)
    return b

if __name__ == "__main__":
    board = buildBoard([[1,0,0,1,1,0],
                        [0,0,0],
                        [1,1,0,0,0],
                        [0,1,1,1,1],
                        [1,1,0,0],
                        [1,0,0,1],
                        [0,1,1,1]])

    printBoard(board)
    m = getmask(4)
    print(cyan, value(board), endc)
    
    
    m = getmask(4)
    zz0 = F.conv2d(board[:,0], m)
    zz1 = F.conv2d(board[:,1], m)
    v1 = 1*(torch.amax(zz0, dim=(0,1,2))/4 >= 1)
    v2 = -1*(torch.amax(zz1, dim=(0,1,2))/4 >= 1)

    print(purple, m, endc)
    print(orange, zz0, endc)
    print(lime, zz1, endc)
    print(cyan, v1, v2, endc)
