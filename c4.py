import torch
import numpy as np
import os, time, numpy as np
from tqdm import trange
from utils import *


def newBoard(shape, device="cpu"):
    if len(shape) == 2: b, h, w = 1, *shape
    elif len(shape) == 3: b, h, w = shape
    return torch.zeros((b, 2, h, w), dtype=torch.float32, device=torch.device(device), requires_grad=False)

def legalActions(boards_:torch.tensor):
    boards = boards_.clone()
    if boards.dim() == 3: boards = boards_.unsqueeze(0)
    boards = torch.sum(boards,axis=1)
    return 1*(boards[:,0,:]==0).squeeze()

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

def drop(boards_:torch.tensor, columns:torch.tensor, color:int):
    boards = boards_.clone()
    batchsize, _, height, width = boards.shape
    iii = height - torch.sum(boards, axis=(1,2), dtype=torch.int32).squeeze(axis=1) - 1
    bidxs = torch.arange(batchsize)
    boards[bidxs,color,iii[bidxs,columns],columns] = 1
    return boards

if __name__ == "__main__":
    #boards = newBoard((3,6,7))
    #actions = [2,6,6]
    #boards = drop(boards, actions, 1)
    #boards = drop(boards, actions, 1)
    #boards = drop(boards, actions, 1)
    #actions = [5,1,2]
    #boards = drop(boards, actions, 0)
    #actions = [0,4,6]
    #boards = drop(boards, actions, 1)
    #actions = [4,6,0]
    #boards = drop(boards, actions, 0)
    #printBoard(boards)

    b = newBoard((5,6,7))
    b = drop(b, [0, 3, 6, 6, 1], 0)
    b = drop(b, [6, 2, 1, 0, 3], 1)
    #b = drop(b, 3, 0)
    #b = drop(b, 3, 0)
    #b = drop(b, 3, 0)
    #b = drop(b, 0, 1)
    #b = drop(b, 1, 1)
    #b = drop(b, 2, 1)
    
    printBoard(b)
