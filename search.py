import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from deepc4.c4_ import *
np.set_printoptions(suppress=True, linewidth=200, precision=4)

class node:
    def __init__(self, board, mask, token, action):
        # token is 1, -1, corresponding to the color of the move that created to this node
        # action is the column that was played to create this node
        self.board = board
        self.token = token
        self.action = action
        self.mask = mask
        self.allowed = np.where(legalMoves(board))[0]
        self.val = value(board, mask)
        self.children = []
        self.is_leaf = True
        self.is_terminal = self.val != 0

    def addKid(self, kid):
        self.children.append(kid)
        self.is_leaf = False

    def findKids(self):
        for i in self.allowed:
            self.addKid(node(drop(self.board, i, -1*self.token), self.mask, -1*self.token, i))

class searcher:
    def __init__(self, boardShape, token, board=None):
        self.current_state = newBoard(boardShape) if board is None else board
        self.boardShape = boardShape
        self.numActions = boardShape[1]
        self.wMask = winMask(boardShape)
        self.token = token
        self.root = node(self.current_state, self.wMask, self.token, None)

    def search(self, depth, currentDepth=0, root:node=None):
        root = self.root if root is None else root
        if currentDepth < depth and not root.is_terminal:
            if root.is_leaf:
                root.findKids()
                root.children = [nod for nod in root.children if nod.value*nod.token == 1]
                    
            for nod in root.children:
                self.search(depth, currentDepth=currentDepth+1, root=nod)
        return root

    def drop(self, col):
        self.observeMove(col, root=self.root)
        return self.current_state

    def observeMove(self, move, root:node=None):
        root = self.root if root is None else root
        for nod in root.children:
            if nod.action == move:
                self.root = nod
                return nod

def playVsSearch(boardShape, depth, seed=None): # the move selection here makes no sense. It needs to recognize when the user can win and stop it
    r = np.random.uniform(0, 1) if seed is None else seed
    userFirst = r < 0.5

    ivern = searcher(boardShape, 1)
    ivern.search(depth)
    ivern.grade()

    board = newBoard(boardShape)
    printBoard(board)
    for turn in range(board.size+1):
        allow = legalMoves(board)
        if np.sum(allow) == 0:
            val = 0
            printBoard(board)
            print(gray, "game drawn", endc)
            return val
        if (not userFirst) == turn%2:
            vals = [nod.val for nod in ivern.root.children]
            [printBoard(nod.board) for nod in ivern.root.children]
            action = ivern.root.children[np.argmax(vals)].action
            board = drop(board, action, 1)
            ivern.observeMove(action)
            ivern.search(depth)

            val = value(board, ivern.wMask)
            if val != 0:
                print(bold, red, "the agent wins!", endc)
                printBoard(board)
                return val
        else:
            msg = f"{blue}enter move {np.where(allow)}:\n{endc}"
            validinp = False
            fails = 0
            while not validinp:
                try:
                    inp = input(msg)
                    action = int(inp)
                    board = drop(board, action, -1)
                    ivern.observeMove(action)
                    ivern.search(depth)
                    validinp = True
                except:
                    fails += 1
                    print(red, f"invalid input. receieved: {inp}(type:{type(inp)})", endc)
                    assert fails < 10, "too many invalid inputs, erroring out"
            val = value(board, ivern.wMask)
            if val != 0:
                print(bold, green, "the user wins!", endc)
                printBoard(board)
                return val

        printBoard(board)
        print()
    return 255


playVsSearch((6, 7), 6)