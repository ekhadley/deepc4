import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import numpy as np
import os, time, numpy as np
from tqdm import trange
from c4 import *
from utils import *


class policy(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, leaky=True, wd=0.0003):
        super(policy, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.leaky = leaky
        self.cuda_ = cuda
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        if leaky:
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()
            self.act3 = nn.LeakyReLU()
        else:
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
        self.lin1 = nn.Linear(32*h*w, 512)
        self.act4 = nn.ReLU()
        
        self.lin2 = nn.Linear(512, 256)
        self.act5 = nn.ReLU()

        self.lin3 = nn.Linear(256, 128)
        self.act6 = nn.ReLU()

        self.lin4 = nn.Linear(128, w)
        self.act7 = nn.Softmax(dim=1)
        
        if self.cuda_: self.to("cuda")

        #self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

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
    def __init__(self, boardSize, lr=.001, stationary=True, color=1, cuda=False, wd=0.003, memSize=5000):
        self.boardSize, self.lr, self.color = boardSize, lr, color
        self.numActions = boardSize[1]
        self.cuda_ = cuda
        self.wd = wd
        
        self.policy = policy(boardSize, lr=lr, stationary=stationary, cuda=cuda, wd=self.wd)
        exst = torch.tensor(np.zeros(shape=(1, 2, *boardSize), dtype=np.float32))
        exal = torch.tensor(np.ones(shape=(self.numActions), dtype=np.float32))
        print(red, bold, type(torchviz.make_dot(self.policy(exst, exal), params=dict(self.policy.named_parameters())))))
        self.policy.forward = torch.jit.trace(self.policy.forward, example_inputs=(exst, exal))

        #self.states, self.allowed, self.actions, self.weights = [], [], [], []
        self.memSize = memSize
        self.recorded_steps = 0
        self.wlogprobs = self.clearMem()
    
    def observe(self, board):
        a = np.where(board==self.color, 1, 0)
        b = np.where(board==-1*self.color, 1, 0)
        if self.cuda_: obs = torch.cuda.FloatTensor(np.float32([a, b]))
        else: obs = torch.tensor(np.float32([a, b]))
        return obs

    def chooseAction(self, state, allowed):
        if isinstance(allowed, np.ndarray): allowed = torch.tensor(allowed)
        if isinstance(state, np.ndarray): state = torch.tensor(state)
        if self.cuda_: state, allowed = state.to("cuda"), allowed.to("cuda")
        if state.ndim == 3: state = state.unsqueeze(0)
        dist = torch.flatten(self.policy(state, allowed))
        action = torch.distributions.Categorical(dist).sample().detach().item()
        return dist, action

    def drop(self, board, col):
        return drop(board, col, self.color)

    def train(self):
        self.policy.zero_grad()
        #print(bold, pink, self.recorded_steps, endc)
        loss = -torch.mean(self.wlogprobs[:self.recorded_steps])
        loss.backward(retain_graph=True)
        self.policy.opt.step()
        self.wlogprobs = self.clearMem()
        self.policy.zero_grad()
        return loss

    def addEp(self, dists, actions, weights, numTurns):
        if isinstance(weights, (list, np.ndarray)): weights = torch.tensor(weights)
        if self.cuda_: weights = weights.to("cuda")
        probs = torch.sum((dists*actions), axis=1)
        logprobs = torch.log(probs)
        wlogprobs = logprobs*weights
        self.wlogprobs[self.recorded_steps:self.recorded_steps+numTurns] += wlogprobs
        self.recorded_steps += numTurns

    def clearMem(self):
        self.recorded_steps = 0
        return torch.zeros((self.memSize), dtype=torch.float32, device="cuda" if self.cuda_ else "cpu")
    def save(self, path, name):
        self.policy.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.policy.load_state_dict(sd)
    def loadPolicy(self, sd):
        self.policy.load_state_dict(sd)
    def stateDict(self):
        return self.policy.state_dict()