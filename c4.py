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

def value(board, connect=4):
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

    diag1 = diags(board, minLen=connect)
    for d in diag1:
        d_p1, d_p2 = 0, 0
        for e in d:
            if e == 1: d_p1 += 1
            if e == -1: d_p2 += 1
            if d_p1 == 4: return 1
            if d_p2 == 4: return -1
    
    diag2 = diags(np.flip(board, axis=0), minLen=connect)
    for d in diag2:
        dt_p1, dt_p2 = 0, 0
        for e in d:
            if e == 1: dt_p1 += 1
            if e == -1: dt_p2 += 1
            if dt_p1 == 4: return 1
            if dt_p2 == 4: return -1
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