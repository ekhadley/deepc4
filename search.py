import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from c4 import *
np.set_printoptions(suppress=True, linewidth=200, precision=4)

class node:
    def __init__(self, parent, board, color, action):
        # color is 1, -1, corresponding to the color of the move that created to this node
        # action is the column that was played to create this node
        self.board = board
        self.parent = parent
        self.color = color
        self.action = action
        self.allowed = np.where(legalActions(board))[0]
        self.val = value(board)
        self.children = []
        self.is_leaf = True
        self.is_terminal = self.val != 0

    def findKids(self):
        v = 0
        for i in self.allowed:
            newstate = drop(self.board, i, -1*self.color)
            kid = node(self, newstate, -1*self.color, i)
            self.children.append(kid)
            v += abs(kid.val)
            self.is_leaf = False
        if v != 0:
            self.value = self.color
            self.children = [kid for kid in self.children if kid.val != 0]
            self.updateParent()

    def updateParent(self):
        # if we are updating a non leaf node, it means one of that node's offspring ends in a win for somebody
        
        self.parent.val = self.color
        self.parent.children = [self]
        self.parent.updateParent()

class searcher:
    def __init__(self, boardShape, color, board=None):
        self.current_state = newBoard(boardShape) if board is None else board
        self.boardShape = boardShape
        self.numActions = boardShape[1]
        self.color = color
        self.root = node(None, self.current_state, self.color, None)

    def search(self, depth, currentDepth=0, root:node=None):
        root = self.root if root is None else root
        if currentDepth < depth and not root.is_terminal:
            if root.is_leaf:
                root.findKids()
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
        allow = legalActions(board)
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

            val = value(board)
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
            val = value(board)
            if val != 0:
                print(bold, green, "the user wins!", endc)
                printBoard(board)
                return val

        printBoard(board)
        print()
    return 255

if __name__ == "__main__":
    playVsSearch((6, 7), 6)