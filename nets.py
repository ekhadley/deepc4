from c4 import legalActions
from utils import *

class PolicyNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, wd=0.0003, adam=False):
        super(PolicyNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.cuda_ = cuda
        self.bias = True
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, padding_mode="zeros")
        self.act1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode="zeros")
        self.act2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode="zeros")
        self.act3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode="zeros")
        self.act4 = nn.LeakyReLU()
        
        self.lin1 = nn.Linear(256*h*w, 1024, bias=self.bias)
        self.act5 = nn.LeakyReLU()
        
        self.lin2 = nn.Linear(1024, 512, bias=self.bias)
        self.act6 = nn.LeakyReLU()
        
        self.lin3 = nn.Linear(512, 256, bias=self.bias)
        self.act7 = nn.LeakyReLU()
        
        self.lin4 = nn.Linear(256, w, bias=self.bias)
        self.act8 = nn.Softmax(dim=1)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, X:torch.Tensor):
        assert X.ndim == 4, bold + red + f"X.shape={X.shape}. should have [batch, 2, height, width]" + endc
        allowed = legalActions(X)
        X = self.act1(self.conv1(X))
        X = self.act2(self.conv2(X))
        X = self.act3(self.conv3(X))
        X = self.act4(self.conv4(X))
        X = X.reshape(X.shape[0], -1)
        X = self.act5(self.lin1(X))
        X = self.act6(self.lin2(X))
        X = self.act7(self.lin3(X))
        X = self.lin4(X)
        X = self.act8(X - 1e16*(1-allowed))
        #X = F.normalize(torch.abs(X*allowed), dim=1, p=1)
        return X
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train_policy(self, states, actions, weights, debug=True):
        dists = self.forward(states)
        probs = torch.sum(dists*actions, dim=1)
        logprobs = torch.log(probs)
        wlogprobs = logprobs*weights
        loss = -torch.mean(wlogprobs)

        if 1: #################################################### debug
            for i in range(len(dists)):
                si, di, ai, wi = fromt(states[i]), fromt(dists[i]), fromt(actions[i]), fromt(weights[i])
                if -np.log(di[np.argmax(ai)]) == np.inf:
                    printBoard(si)
                    print(f"{red}dists[{i}]={di}")
                    print(f"{lemon}actions[{i}]={ai}")
                    print(f"{blue}weights[{i}]={wi}")
                    print(f"{orange}aprob = {di[np.argmax(ai)]}")
                    print(f"{orange}ln(aprob) = {np.log(di[np.argmax(ai)])}{endc}")
                    print(bold, underline, "======================================================", endc)
        if debug:
            print(green, f"dists=\n{dists}", endc)
            print(gray, f"H(probs)={torch.mean(-logprobs).item():.4f}" + endc)
            print(pink, f"{logprobs=}", endc)
            print(orange, f"{weights=}", endc)
            print(red, f"{probs=}", endc)
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

class ValueNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, wd=0.0003, adam=True):
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

        #self.forward = torch.jit.trace(self.forward, torch.zeros((1, 2, *boardShape), dtype=torch.float32, device=torch.device("cuda") if cuda else torch.device("cpu")))

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

    def train_valnet(self, states, outcomes, debug=True):
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

class tinyPolicyNet(nn.Module):
    def __init__(self, boardShape, lr=.001, stationary=True, cuda=False, wd=0.0003, adam=False):
        super(tinyPolicyNet, self).__init__()
        self.boardShape, self.lr = boardShape, lr
        self.cuda_ = cuda
        self.bias = True
        h, w = boardShape

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1, padding_mode="zeros")
        self.act1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, padding_mode="zeros")
        self.act2 = nn.LeakyReLU()
        
        self.lin1 = nn.Linear(32*h*w, 512, bias=self.bias)
        self.act3 = nn.LeakyReLU()
        
        self.lin2 = nn.Linear(512, w, bias=self.bias)
        self.act4 = nn.Softmax(dim=1)

        if self.cuda_: self.to("cuda")

        if adam: self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.99, 0.99) if stationary else None)
        else: self.opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, X:torch.Tensor):
        assert X.ndim == 4, bold + red + f"X.shape={X.shape}. should have [batch, 2, height, width]" + endc
        allowed = legalActions(X)
        X = self.act1(self.conv1(X))
        X = self.act2(self.conv2(X))
        X = X.reshape(X.shape[0], -1)
        X = self.act3(self.lin1(X))
        X = self.lin2(X)
        X = self.act4(X - 1e16*(1-allowed))
        #X = F.normalize(torch.abs(X*allowed), dim=1, p=1)
        return X
    def __call__(self, X:torch.Tensor): return self.forward(X)

    def train_policy(self, states, actions, weights, debug=True):
        dists = self.forward(states)
        probs = torch.sum(dists*actions, dim=1)
        logprobs = torch.log(probs)
        wlogprobs = logprobs*weights
        loss = -torch.mean(wlogprobs)

        if 0: #################################################### debug
            for i in range(len(dists)):
                si, di, ai, wi = fromt(states[i]), fromt(dists[i]), fromt(actions[i]), fromt(weights[i])
                if -np.log(di[np.argmax(ai)]) == np.inf:
                    printBoard(si)
                    print(f"{red}dists[{i}]={di}")
                    print(f"{lemon}actions[{i}]={ai}")
                    print(f"{blue}weights[{i}]={wi}")
                    print(f"{orange}aprob = {di[np.argmax(ai)]}")
                    print(f"{orange}ln(aprob) = {np.log(di[np.argmax(ai)])}{endc}")
                    print(bold, underline, "======================================================", endc)
        if debug:
            print(green, f"dists=\n{dists}", endc)
            print(gray, f"H(probs)={torch.mean(-logprobs).item():.4f}" + endc)
            print(pink, f"{logprobs=}", endc)
            print(orange, f"{weights=}", endc)
            print(red, f"{probs=}", endc)
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