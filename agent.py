import torch
import torch.nn as nn
import numpy as np
import os, time, numpy as np
from tqdm import trange
from c4 import *
from utils import *


class policy(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, leaky=True):
        super(policy, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.leaky = leaky
        self.cuda_ = cuda
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        if leaky:
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()
            self.act3 = nn.LeakyReLU()
        else:
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
        self.lin1 = nn.Linear(64*h*w, 512)
        self.act4 = nn.ReLU()
        
        self.lin2 = nn.Linear(512, 256)
        self.act5 = nn.ReLU()

        self.lin3 = nn.Linear(256, 128)
        self.act6 = nn.ReLU()

        self.lin4 = nn.Linear(128, w)
        self.act7 = nn.Softmax(dim=1)
        
        if self.cuda_: self.to("cuda")

        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=.006, betas=(0.99, 0.99) if stationary else None)
        #self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=.006)

    def forward(self, X:torch.Tensor, allowed:torch.Tensor):
        if self.cuda_: X, allowed = X.to("cuda"), allowed.to("cuda")
        X = self.act1(self.conv1(X))
        X = self.act2(self.conv2(X))
        X = self.act3(self.conv3(X))
        X = X.reshape(X.shape[0], -1)
        X = self.act4(self.lin1(X))
        X = self.act5(self.lin2(X))
        X = self.act6(self.lin3(X))
        X = self.lin4(X)
        X += (1e12*(allowed-1)) # puts the pre-softmax values of illegal moves to -inf
        X = self.act7(X)
        return X
    def __call__(self, X:torch.Tensor, allowed:torch.Tensor): return self.forward(X, allowed)
    
    def train(self, states, allowed, actions, weights):
        if self.cuda_:
            states, allowed, actions, weights = states.to("cuda"), allowed.to("cuda"), actions.to("cuda"), weights.to("cuda")
        probs = self.forward(states, allowed)
        loss = self.loss(probs, actions, weights)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def loss(self, probs, actions, weights, debug=False):
        aprobs = torch.sum((probs*actions), axis=1)
        logprobs = torch.log(aprobs)
        if debug:
            print(yellow, "probs:", probs.shape, endc)
            print(red, "actions:", actions.shape, endc)
            print(green, "weights:", weights.shape, endc)
            print(orange, "aprobs:", aprobs, endc)
            print(blue, "logprobs:", logprobs, endc)
        return -torch.mean(logprobs*weights)

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")
    def load(self, path):
        self.load_state_dict(torch.load(path))

class vpoAgent:
    def __init__(self, boardSize, lr=.001, stationary=True, color=1, cuda=False):
        self.boardSize, self.lr, self.color = boardSize, lr, color
        self.numActions = boardSize[1]
        self.cuda_ = cuda
        
        self.policy = policy(boardSize, lr=lr, stationary=stationary, cuda=cuda)
        exst = torch.tensor(np.zeros(shape=(1, 2, *boardSize), dtype=np.float32))
        exal = torch.tensor(np.ones(shape=(self.numActions), dtype=np.float32))
        self.policy.forward = torch.jit.trace(self.policy.forward, example_inputs=(exst, exal))

        self.states, self.allowed, self.actions, self.weights = [], [], [], []
    
    def observe(self, board):
        a = np.where(board==self.color, 1, 0)
        b = np.where(board==-1*self.color, 1, 0)
        obs = np.float32([a, b])
        return obs

    def chooseAction(self, state, allowed):
        if isinstance(allowed, np.ndarray): allowed = torch.tensor(allowed)
        if isinstance(state, np.ndarray): state = torch.tensor(state)
        if self.cuda_: state, allowed = state.to("cuda"), allowed.to("cuda")
        if state.ndim == 3: state = state.unsqueeze(0)
        dist = self.policy(state, allowed)
        dist = torch.distributions.Categorical(dist)
        action = dist.sample()
        return dist, action

    def drop(self, board, col):
        return drop(board, col, self.color)

    def train(self):
        states = torch.tensor(np.float32(self.states))
        allowed = torch.tensor(np.float32(self.allowed))
        actions = torch.tensor(np.float32(self.actions))
        weights = torch.tensor(np.float32(self.weights))
        return self.policy.train(states, allowed, actions, weights)

    def remember(self, states, allowed, actions, weights):
        self.states += states
        self.allowed += allowed
        self.actions += actions
        self.weights += weights
    def forget(self):
        self.states, self.allowed, self.actions, self.weights = [], [], [], []

    def save(self, path, name):
        self.policy.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.policy.load_state_dict(sd)
    def loadPolicy(self, sd):
        self.policy.load_state_dict(sd)
    def stateDict(self):
        return self.policy.state_dict()