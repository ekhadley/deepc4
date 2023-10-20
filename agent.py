from utils import *
from nets import *
from c4 import drop

class vpoAgent:
    def __init__(self, boardShape, color, memSize=10_000, vlr=0.001, plr=.001, stationary=True, cuda=False, wd=0.003, adam=False):
        self.boardShape, self.color = boardShape, color
        self.plr, self.vlr = plr, vlr
        self.numActions = boardShape[1]
        self.cuda_ = cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.wd = wd
        self.memSize = memSize
        
        #self.policy = policyNet(boardShape, lr=plr, stationary=stationary, cuda=cuda, wd=self.wd, adam=adam)
        self.policy = tinyPolicyNet(boardShape, lr=plr, stationary=stationary, cuda=cuda, wd=self.wd, adam=adam)
        self.valnet = ValueNet(boardShape, lr=vlr, stationary=stationary, cuda=cuda, wd=self.wd, adam=adam)

        self.dist_ = torch.distributions.Categorical(torch.tensor([1/self.numActions]*self.numActions))

        self.states, self.actions, self.weights, self.recorded_steps = self.getBlankMem()
    
    @torch.no_grad()
    def chooseAction(self, state:torch.Tensor):
        if state.ndim == 3: state = state.unsqueeze(0)
        dist = self.policy(state)
        cat = torch.distributions.Categorical(dist)
        action = cat.sample()
        return dist, action

    def drop(self, board, col):
        return drop(board, col, self.color)
    
    @torch.no_grad()
    def observe(self, board):
        assert board.ndim == 4, f"board.shape={board.shape}, expected ndim=4"
        if self.color: return torch.flip(board, dims=(1,))
        return board
    
    def train(self):
        #states = self.states[:self.recorded_steps]
        p = torch.randperm(self.recorded_steps)
        states = self.states[:self.recorded_steps][p]
        actions = self.actions[:self.recorded_steps][p]
        outcomes = self.weights[:self.recorded_steps][p]
        weights = self.valnet(states).detach()
        _, vloss, acc = self.valnet.train_valnet(states, outcomes)
        ploss = self.policy.train_policy(states, actions, outcomes) # uses the unweighted outcomes
        #ploss = self.policy.train_policy(states, actions, weights) # uses the values from the valnet to weight
        #ploss = self.policy.train_policy(states, actions, outcomes-weights) # uses the advantage 
        self.resetMem()
        return vloss, ploss, acc

    @torch.no_grad()
    def addGame(self, states, actions, outcome, numTurns):
        self.states[self.recorded_steps:self.recorded_steps+numTurns] = states
        self.actions[self.recorded_steps:self.recorded_steps+numTurns] = actions
        self.weights[self.recorded_steps:self.recorded_steps+numTurns] = outcome
        self.recorded_steps += numTurns

    def resetMem(self):
        self.states, self.actions, self.weights, self.recorded_steps = self.getBlankMem()
    def getBlankMem(self):
        yield torch.zeros((self.memSize, 2, *self.boardShape), dtype=torch.float32, device=self.device, requires_grad=False)
        yield torch.zeros((self.memSize, self.numActions), dtype=torch.float32, device=self.device, requires_grad=False)
        yield torch.zeros((self.memSize), dtype=torch.float32, device=self.device, requires_grad=False)
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

