import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from c4 import drop, legalActions
from utils import *

class vpoAgent:
    def __init__(self, boardShape, color, memSize=10_000, vlr=0.001, plr=.001, stationary=True, cuda=False, wd=0.003, adam=False):
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
    
    @torch.no_grad()
    def chooseAction(self, state:torch.Tensor):
        if state.ndim == 3: state = state.unsqueeze(0)
        dist = torch.log(self.policy(state))
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
        weights = self.valnet(states).detach()
        _, vloss, acc = self.valnet.train_(states, outcomes)
        ploss = self.policy.train_(states, actions, outcomes) # uses the unweighted outcomes
        #ploss = self.policy.train_(states, actions, weights) # uses the values from the valnet to weight
        #ploss = self.policy.train_(states, actions, outcomes-weights) # uses the advantage 
        self.resetMem()
        return vloss, ploss, acc

    @torch.no_grad()
    def addGame(self, states, actions, outcome, numTurns):
        if self.recorded_steps + numTurns >= self.memSize:
            assert 0
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
        assert 0
        sd = torch.load(path)
        self.net.load_state_dict(sd)
    def loadPolicy(self, sd):
        self.policy.load_state_dict(sd)
    def loadValnet(self, sd):
        self.valnet.load_state_dict(sd)
    def policyStateDict(self):
        return self.policy.state_dict()

class ValueNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, wd=0.0003, adam=False):
        super(ValueNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.cuda_ = cuda
        self.bias = True
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.LeakyReLU()

        self.lin1 = nn.Linear(256*h*w, 1024, bias=self.bias)
        self.bn5 = nn.BatchNorm1d(1024)
        self.act5 = nn.LeakyReLU()

        self.lin2 = nn.Linear(1024, 512, bias=self.bias)
        self.bn6 = nn.BatchNorm1d(512)
        self.act6 = nn.LeakyReLU()

        self.lin3 = nn.Linear(512, 256, bias=self.bias)
        self.bn7 = nn.BatchNorm1d(256)
        self.act7 = nn.LeakyReLU()

        self.lin4 = nn.Linear(256, 1, bias=self.bias)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        self.eval()

    def forward(self, X:torch.Tensor):
        if self.cuda_ and not X.is_cuda: X = X.to("cuda"); print(bold, lemon, "non cuda passed to cuda model, implicit move op", endc)
        X = self.act1(self.bn1(self.conv1(X)))
        X = self.act2(self.bn2(self.conv2(X)))
        X = self.act3(self.bn3(self.conv3(X)))
        X = self.act4(self.bn4(self.conv4(X)))
        X = X.reshape(X.shape[0], -1)
        X = self.act5(self.bn5(self.lin1(X)))
        X = self.act6(self.bn6(self.lin2(X)))
        X = self.act7(self.bn7(self.lin3(X)))
        X = self.lin4(X)
        return X.squeeze()
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train_(self, states, outcomes, debug=True):
        self.train()
        vals = self.forward(states)
        loss = F.mse_loss(vals, outcomes)
        acc = torch.mean((torch.sign(vals)==torch.sign(outcomes)).float()).item()
        
        if 0: #################################################### debug
            for i in range(len(vals)):
                printBoard(states[i])
                print(f"{red}{vals[i]=}")
                print(f"{blue}{outcomes[i]=}")
                print()
        if debug:
            print(purple, f"\n\n{outcomes=}", endc)
            print(blue, f"{vals=},{bold} [{green}{(torch.sum(1*(vals>0))/len(outcomes)).item():.3f},{red}{(torch.sum(1*(vals<0))/len(outcomes)).item():.3f}]{gray}({len(outcomes)})", endc)
            print(lime, f"acc={acc:.4f}")
            print(cyan, f"value_loss={loss.item()}", endc)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eval()
        return vals, loss.item(), acc

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")
    def load(self, path):
        self.load_state_dict(torch.load(path))

class PolicyNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, wd=0.0003, adam=False):
        super(PolicyNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.cuda_ = cuda
        self.bias = True
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode="zeros")
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.LeakyReLU()
        
        self.lin1 = nn.Linear(256*h*w, 1024, bias=self.bias)
        self.bn5 = nn.BatchNorm1d(1024)
        self.act5 = nn.LeakyReLU()
        
        self.lin2 = nn.Linear(1024, 512, bias=self.bias)
        self.bn6 = nn.BatchNorm1d(512)
        self.act6 = nn.LeakyReLU()
        
        self.lin3 = nn.Linear(512, 256, bias=self.bias)
        self.bn7 = nn.BatchNorm1d(256)
        self.act7 = nn.LeakyReLU()
        
        self.lin4 = nn.Linear(256, w, bias=self.bias)
        self.act8 = nn.Softmax(dim=1)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        self.eval()

    def forward(self, X:torch.Tensor):
        assert X.ndim == 4, bold + red + f"X.shape={X.shape}. should have [batch, 2, height, width]" + endc
        allowed = legalActions(X)
        X = self.act1(self.bn1(self.conv1(X)))
        X = self.act2(self.bn2(self.conv2(X)))
        X = self.act3(self.bn3(self.conv3(X)))
        X = self.act4(self.bn4(self.conv4(X)))
        X = X.reshape(X.shape[0], -1)
        X = self.act5(self.bn5(self.lin1(X)))
        X = self.act6(self.bn6(self.lin2(X)))
        X = self.act7(self.bn7(self.lin3(X)))
        X = self.lin4(X)
        #print(bold, gray, X, endc)
        X = self.act8(X - 1e16*(1-allowed))
        #X = F.normalize(torch.abs(X*allowed), dim=1, p=1)
        #print(bold, green, X, endc)
        return X
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train_(self, states, actions, weights, debug=True):
        self.train()
        dists = self.forward(states)
        probs = torch.sum(dists*actions, dim=1)
        logprobs = torch.log(probs)
        wlogprobs = logprobs*weights
        loss = -torch.mean(weights*probs)

        if 0: #################################################### debug
            for i in range(len(dists)):
                printBoard(states[i])
                print(f"{red}dists[{i}]={dists[i].cpu().detach().numpy()}")
                print(f"{lemon}actions[{i}]={actions[i].cpu().detach().numpy()}")
                print(f"{blue}weights[{i}]={weights[i].cpu().detach().numpy()}")
                print()
            print(bold, underline, "======================================================", endc)
        if debug:
            print(green, f"dists=\n{dists}", endc)
            print(gray, f"H(probs)={torch.mean(-logprobs).item():.4f}" + endc)
            print(pink, f"{logprobs=}", endc)
            print(orange, f"{weights=}", endc)
            #print(lemon, f"{wlogprobs=}", endc)
            print(cyan, f"policy_loss={loss.item()}", endc)
            #plt.hist(weights.detach().numpy(), bins=100)
            #plt.show()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eval()
        return loss.item()

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")
    def load(self, path):
        self.load_state_dict(torch.load(path))