import torch
import numpy as np
import os, time, numpy as np
from tqdm import trange
from utils import *


def newBoard(size):
    return np.zeros(size)

def copyBoard(board): return np.array(board, copy=True)

def drop(board, col, val):
    h, w = board.shape
    assert 0 <= col < w
    for i in range(h):
        if board[h-i-1][col] == 0:
            nboard = copyBoard(board)
            nboard[h-i-1][col] = val
            return nboard
    return copyBoard(board)

def legalMoves(board):
    al = 1*(board[0,:]==0)
    return torch.tensor(al), al

def randomMove(board):
    moves = legalMoves(board, mask=False)
    if len(moves) >= 1: return np.random.choice(np.where(moves))
    return -1

def diags(board, minLen=1):
    h, w = board.shape
    return [np.diagonal(board, i) for i in range(-h+minLen, w-minLen+1)]

def value_(board, connect=4):
    bh, bw = board.shape
    h_p1, v_p1 = 0, 0
    h_p2, v_p2 = 0, 0
    for i in range(board.size):
        hcx, hcy = i%bw, i//bw
        if hcx == 0: h_p1=0; h_p2=0
        hval = board[hcy][hcx]
        if hval == 1: h_p1 += 1
        else: h_p1 = 0
        if hval == -1: h_p2 += 1
        else: h_p2 = 0
        if h_p1 == 4: return 1
        if h_p2 == 4: return -1

        vcx, vcy = i//bh, i%bh
        if vcy == 0: v_p1=0; v_p2=0
        vval = board[vcy][vcx]
        if vval == 1: v_p1 += 1
        else: v_p1 = 0
        if vval == -1: v_p2 += 1
        else: v_p2 = 0
        if v_p1 == 4: return 1
        if v_p2 == 4: return -1

    for didx in range(-bh+connect, bw-connect+1):
        d = np.diagonal(board, didx)
        d_p1, d_p2 = 0, 0
        for e in d:
            if e == 1: d_p1 += 1
            if e == -1: d_p2 += 1
            if d_p1 == 4: return 1
            if d_p2 == 4: return -1
    
    bflip = np.flip(board, axis=0)
    for didx in range(-bw+connect, bh-connect+1):
        d = np.diagonal(bflip, didx)
        d_p1, d_p2 = 0, 0
        for e in d:
            if e == 1: d_p1 += 1
            if e == -1: d_p2 += 1
            if d_p1 == 4: return 1
            if d_p2 == 4: return -1
    return 0

def winMask(shape, cnct=4):
    h, w = shape
    #numMasks = h*(w-cnct+1) + w*(h-cnct+1) + (abs(-h+cnct - w-cnct+1)) + (abs(-w+cnct - h-cnct+1))
    numMasks = h*(w-cnct+1) + w*(h-cnct+1) + 2*(h-cnct+1)*(w-cnct+1)
    masks = np.zeros((numMasks, h, w))
    count = 0
    for r in range(h): # finds all horizontal connect masks
        for c in range(w-cnct+1):
            p = np.ones((1, cnct))
            m = np.pad(p, ((r,h-r-1),(c, w-c-cnct)), constant_values=False)
            masks[count] += m
            count += 1
    
    for r in range(h-cnct+1): # finds all vertical connect masks
        for c in range(w):
            p = np.ones((cnct, 1))
            m = np.pad(p, ((r,h-r-cnct),(c,w-c-1)), constant_values=False)
            masks[count] += m
            count += 1

    for r in range(h-cnct+1): # finds all diagonal masks
        for c in range(w-cnct+1):
            d = np.eye(cnct)
            m = np.pad(d, ((r,h-r-cnct),(c, w-c-cnct)), constant_values=False)
            masks[count] += m
            masks[count+1] += np.flip(m, axis=1)
            count += 2

    return masks

def value(board, mask=None):
    mask = winMask(board.shape) if mask is None else mask
    val1 = np.sum(board*mask, axis=(1,2)) > 3
    val2 = np.sum(board*mask, axis=(1,2)) < -3
    v1, v2 = np.sum(val1), np.sum(val2)
    if v1: return 1
    if v2: return -1
    return 0

def observe(board):
    a = np.where(board==1, 1, 0)
    b = np.where(board==-1, 1, 0)
    obs = torch.tensor(np.float32([a, b]))
    print(orange, bold, board, endc)
    print(green, bold, obs, endc)
    return obs

def printBoard(board):
    str = gray
    for i in range(board.shape[1]):
        str += f" {i} "
    str += "\n"
    for row in board:
        for col in row:
            if col == 1: str += green + " O "
            elif col == -1: str += red + " X "
            else: str += gray + " . "
        str += "\n"
    str += endc
    print(str)

