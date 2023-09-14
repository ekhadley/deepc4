import torch
import torch.nn as nn
import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from c4 import drop, legalActions
from utils import *


class ValueNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, leaky=True, wd=0.0003, adam=False):
        super(ValueNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.leaky = leaky
        self.cuda_ = cuda
        self.bias = True
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        if self.leaky:
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()
            self.act3 = nn.LeakyReLU()
            self.act4 = nn.LeakyReLU()
        else:
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
            self.act3 = nn.Tanh()
            self.act4 = nn.Tanh()

        self.lin1 = nn.Linear(256*h*w, 1024, bias=self.bias)
        self.act5 = nn.ReLU()
        self.lin2 = nn.Linear(1024, 512, bias=self.bias)
        self.act6 = nn.ReLU()
        self.lin3 = nn.Linear(512, 256, bias=self.bias)
        self.act7 = nn.ReLU()
        self.lin4 = nn.Linear(256, 1, bias=self.bias)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, X:torch.Tensor):
        if self.cuda_ and not X.is_cuda: X = X.to("cuda"); print(bold, lemon, "non cuda passed to cuda model, implicit move op", endc)
        X = self.act1(self.conv1(X))
        X = self.act2(self.conv2(X))
        X = self.act3(self.conv3(X))
        X = self.act4(self.conv4(X))
        X = X.reshape(X.shape[0], -1)
        X = self.act5(self.lin1(X))
        X = self.act6(self.lin2(X))
        X = self.act7(self.lin3(X))
        X = self.lin4(X)
        return X.squeeze()
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train(self, states, outcomes, debug=True):
        vals = self.forward(states)
        loss = F.mse_loss(vals, outcomes)
        acc = torch.mean((torch.sign(vals)==torch.sign(outcomes)).float()).item()
        if debug:
            print(lime, f"\n\nacc={acc:.4f}")
            print(blue, f"{vals=},{bold} [{green}{(torch.sum(1*(vals>0))/len(outcomes)).item():.3f},{red}{(torch.sum(1*(vals<0))/len(outcomes)).item():.3f}]{gray}({len(outcomes)})", endc)
            print(purple, f"{outcomes=}", endc)
            print(cyan, f"value_loss={loss.item()}", endc)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return vals.detach(), loss.item(), acc

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")
    def load(self, path):
        self.load_state_dict(torch.load(path))

class PolicyNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, leaky=True, wd=0.0003, adam=False):
        super(PolicyNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.leaky = leaky
        self.cuda_ = cuda
        self.bias = False
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        if self.leaky:
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()
            self.act3 = nn.LeakyReLU()
            self.act4 = nn.LeakyReLU()
        else:
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
            self.act3 = nn.Tanh()
            self.act4 = nn.Tanh()

        self.lin1 = nn.Linear(256*h*w, 1024, bias=self.bias)
        self.act5 = nn.ReLU()
        self.lin2 = nn.Linear(1024, 512, bias=self.bias)
        self.act6 = nn.ReLU()
        self.lin3 = nn.Linear(512, 256, bias=self.bias)
        self.act7 = nn.ReLU()
        self.lin4 = nn.Linear(256, w, bias=self.bias)
        self.act8 = nn.ReLU()
        self.act9 = nn.Softmax(dim=1)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, X:torch.Tensor):
        if self.cuda_ and not X.is_cuda: X = X.to("cuda"); print(bold, lemon, "non cuda passed to cuda model, implicit move op", endc)
        allowed = legalActions(X)
        X = self.act1(self.conv1(X))
        X = self.act2(self.conv2(X))
        X = self.act3(self.conv3(X))
        X = self.act4(self.conv4(X))
        X = X.reshape(X.shape[0], -1)
        X = self.act5(self.lin1(X))
        X = self.act6(self.lin2(X))
        X = self.act7(self.lin3(X))
        X = self.act8(self.lin4(X))
        X = self.act9(X + 1e16*(allowed-1))
        return X
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train(self, states, actions, weights, debug=True):
        dists = self.forward(states)
        probs = torch.sum(dists*actions, dim=1)
        logprobs = torch.log(probs)
        wlogprobs = weights*logprobs
        loss = -torch.mean(wlogprobs)

        if 0: #################################################### debug
            for i in range(len(dists)):
                printBoard(states[i])
                print(f"{red}{dists[i]=}")
                print(f"{lemon}{actions[i]=}")
                print(f"{blue}{weights[i]=}")
            print(bold, underline, "======================================================", endc)

        if debug:
            print(green, f"{dists=}", endc)
            print(gray, f"H(probs)={torch.mean(-logprobs).item():.4f}" + endc)
            print(pink, f"{logprobs=}", endc)
            print(orange, f"{weights=}", endc)
            print(lemon, f"{wlogprobs=}", endc)
            print(cyan, f"policy_loss={loss.item()}", endc)
            #plt.hist(weights.detach().numpy(), bins=100)
            #plt.show()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")
    def load(self, path):
        self.load_state_dict(torch.load(path))

class vpoAgent:
    def __init__(self, boardShape, color, memSize=15000, vlr=0.001, plr=.001, stationary=True, cuda=False, wd=0.003, adam=False):
        self.boardShape, self.color = boardShape, color
        self.plr, self.vlr = plr, vlr
        self.numActions = boardShape[1]
        self.cuda_ = cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.wd = wd
        self.memSize = memSize
        
        self.policy = PolicyNet(boardShape, lr=plr, stationary=stationary, cuda=cuda, wd=self.wd, adam=adam)
        self.valnet = ValueNet(boardShape, lr=vlr, stationary=stationary, cuda=cuda, wd=self.wd, adam=adam)

        self.dist_ = torch.distributions.Categorical(torch.tensor([1/self.numActions]*self.numActions))

        self.states, self.actions, self.weights, self.recorded_steps = self.getBlankMem()
    
    def chooseAction(self, state:torch.Tensor):
        if state.ndim == 3: state = state.unsqueeze(0)
        dist = self.policy(state)
        with torch.no_grad():
            # hacky fix. how to sample any batch size output without recreating the categorical class?
            try:
                self.dist_.probs = dist
                action = self.dist_.sample()
            except RuntimeError:
                cat = torch.distributions.Categorical(dist)
                action = cat.sample()
        return dist, action

    def drop(self, board, col):
        return drop(board, col, self.color)
    
    def observe(self, board):
        assert board.ndim == 4
        if self.color: return torch.flip(board, dims=(1,))
        return board

    def train(self):
        states = self.states[:self.recorded_steps]
        actions = self.actions[:self.recorded_steps]
        outcomes = self.weights[:self.recorded_steps]
        weights, vloss, acc = self.valnet.train(states, outcomes)

        if 1:
            ploss = self.policy.train(states, actions, weights) # uses the values from the valnet to weight
        else:
            ploss = self.policy.train(states, actions, outcomes) # uses the unweighted outcomes

        self.resetMem()
        return vloss, ploss, acc

    def addGame(self, states, actions, outcome, numTurns):
        if self.recorded_steps + numTurns >= self.memSize:
            self.states = torch.cat((self.states, torch.zeros((self.memSize, 2, *self.boardShape), dtype=torch.float32, device=self.device)))
            self.memSize *= 2
        self.states[self.recorded_steps:self.recorded_steps+numTurns] += states
        self.actions[self.recorded_steps:self.recorded_steps+numTurns] += actions
        self.weights[self.recorded_steps:self.recorded_steps+numTurns] += outcome

        self.recorded_steps += numTurns

    def resetMem(self):
        self.states, self.actions , self.weights, self.recorded_steps = self.getBlankMem()
    def getBlankMem(self):
        yield torch.zeros((self.memSize, 2, *self.boardShape), dtype=torch.float32, device=self.device)
        yield torch.zeros((self.memSize, self.numActions), dtype=torch.float32, device=self.device)
        yield torch.zeros((self.memSize), dtype=torch.float32, device=self.device)
        yield 0

    def save(self, path, name):
        self.policy.save(path, f"p{name}")
        self.valnet.save(path, f"v{name}")
    def load(self, path):
        sd = torch.load(path)
        self.net.load_state_dict(sd)
    def loadPolicy(self, sd):
        self.policy.load_state_dict(sd)
    def loadValnet(self, sd):
        self.valnet.load_state_dict(sd)
    def policyStateDict(self):
        return self.policy.state_dict()