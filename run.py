import torch
import torch.nn as nn
import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from c4 import *
from agent import vpoAgent
import wandb
np.set_printoptions(suppress=True)


def playAgentGame(learner:vpoAgent, opponent:vpoAgent, boardSize, recordExperience=True, show=False, seed=None):
    r = np.random.uniform(0, 1) if seed is None else seed
    if r >= 0.5:
        players = [learner, opponent]
        learnerFirst = True
    else:
        players = [opponent, learner]
        learnerFirst = False

    board = newBoard(boardSize)
    states, allowed, actions = [], [], []
    for turn in range(board.size+1):
        player = players[turn%2]
        allowT, allow = legalMoves(board)
        if torch.sum(allowT) == 0: val = 0; break
        state = player.observe(board)
        dist, action = player.chooseAction(state, allowT)
        board = player.drop(board, action)
        val = value(board)
        if show:
            print(gray, dist.probs.detach().numpy()[0], endc)
            printBoard(board)
            print()
        
        if recordExperience and (not learnerFirst)==turn%2:
            states.append(state)
            allowed.append(allow)
            hot = np.eye(player.numActions)[action]
            actions.append(hot)
        if val != 0: break
    if recordExperience: return states, allowed, actions, turn, val, dist
    return turn, val

def playWithUser(agent, seed=None):
    if isinstance(agent, str):
        agent = vpoAgent((6,7), lr=0.001, stationary=True, color=1, cuda=False)
        agent.load(agent)

    r = np.random.uniform(0, 1) if seed is None else seed
    userFirst = (r >= 0.5)

    board = newBoard(agent.boardSize)
    printBoard(board)
    for turn in range(board.size+1):
        allowT, allow = legalMoves(board)
        if torch.sum(allowT) == 0:
            val = 0
            printBoard(board)
            print(gray, "game drawn", endc)
            return val
        if (not userFirst) == turn%2:
            state = agent.observe(board)
            dist, action = agent.chooseAction(state, allowT)
            board = agent.drop(board, action)
            print(f"{gray}{dist.probs.detach().numpy()[0]}{endc}")
            val = value(board)
            if val != 0:
                print(bold, red, "the agent wins!", endc)
                printBoard(board)
                return val
        else:
            msg = f"{blue}enter move {np.where(allow)}:\n{endc}"
            validinp = False
            while not validinp:
                try:
                    inp = int(input(msg))
                    board = drop(board, inp, -1)
                    validinp = True
                except:
                    print(red, f"invalid input. receieved: {inp}(type:{type(inp)})", endc)
            val = value(board)
            if val != 0:
                print(bold, green, "the user wins!", endc)
                printBoard(board)
                return val

        printBoard(board)
        print()
    return -255

def train(saveDir,
          numGames=1_000_000,
          opponentSamplingWeight=2,
          trainEvery=35,
          discount=0.8,
          valueScale=1,
          testEvery=300,
          examineEvery=10_000,
          lr=0.001,
          numTestGames=130,
          wrThresh=0.6,
          boardSize=(6,7),
          showTestGames=True,
          cuda=False):

    r = vpoAgent(boardSize, lr=lr, stationary=True, color=1, cuda=cuda)
    r.save(saveDir, "0") # save a random policy to be the first opponent

    learner = vpoAgent(boardSize, lr=lr, stationary=True, color=1, cuda=cuda) # initialize the first agent to be trained
    opponent = vpoAgent(boardSize, lr=lr, stationary=True, color=-1, cuda=cuda) # initialize the first opponent to be trained against

    config = {"modelShape": learner.policy, "numGames":numGames, "opponentSamplingWeight":opponentSamplingWeight, "trainEvery":trainEvery, "discount":discount, "testEvery":testEvery, "lr":lr, "numTestGames":numTestGames, "wrThresh":wrThresh, "boardSize":boardSize, "showTestGames":showTestGames, "optimizer":learner.policy.opt.state_dict(), "leaky_acts":learner.policy.leaky, "valueScale":valueScale}
    wandb.init(project="vpoc4", config=config, dir="D:\\wgmn\\wandb\\vpoc4")
    wandb.watch(learner.policy, log="all", log_freq=10)

    beatBest, matchWR, loss = False, 0, 0
    opponents = [r.stateDict()] # we keep the previous best models' state_dicts
    for game in (t:=trange(numGames, ncols=140, unit="games")):
        desc = blue + bold
        opIdx = sampleOpponents(len(opponents), weight=opponentSamplingWeight)
        opponent.loadPolicy(opponents[opIdx])
        desc += f"VSing agent#{opIdx}/{len(opponents)-1}"
        
        states, allowed, actions, numTurns, val, dist = playAgentGame(learner, opponent, boardSize, show=False)
        #desc += f"{gray}{dist.probs.detach().numpy()[0]}{endc}"
        
        nt = len(actions)
        weights = [valueScale*val*discount**(nt-i-1) for i in range(nt)] # discounted weights for each state based on game outcome
        learner.remember(states, allowed, actions, weights)
        
        if game > 0 and game%trainEvery == 0:
            loss = learner.train()
            learner.forget()
        desc += f"{orange}, loss:{loss:.4f}"

        if game > 0 and game%testEvery == 0: # play test match to see if learner has surpassed prev best policy
            score = 0
            opponent.loadPolicy(opponents[-1])
            for midx in (mrange:=trange(numTestGames, ncols=80, unit="games", ascii=" >=")):
                numTurns, val_ = playAgentGame(learner, opponent, boardSize, recordExperience=False, show=showTestGames, seed=midx/numTestGames)
                score += val_
                matchWR = (score/(midx+1) + 1)/2
                beatBest = matchWR >= wrThresh
                mrange.set_description((green if beatBest else red) + f"{matchWR:.3f} vs #{len(opponents)-1}")
            if beatBest:
                opponents.append(learner.stateDict())
                learner.save(saveDir, f"{game}")
        if beatBest: desc += f"{green}, beat with wr={matchWR:.4f}"
        else: desc += f"{red}, lost with wr={matchWR:.4f}"

        if game > 0 and game%examineEvery == 0: #if each model is a bit better than the last, we should see a trend of improving match scores as we vs earlier agents
            scores = [0 for i in range(len(opponents))]
            for op in range(len(opponents)):
                opponent.loadPolicy(opponents[op])
                for _ in range(numTestGames):
                    numTurns_, val_ = playAgentGame(learner, opponent, boardSize, recordExperience=False, show=showTestGames)
                    scores[op] += val_
            playWithUser(learner)
            plt.plot(scores)
            plt.show()

        wandb.log({"gameLength": numTurns, "loss":loss, "matchWR":matchWR, "numAgents":len(opponents), "score":val})
        t.set_description(desc + purple)

save = "D:\\deepc4\\vpo"


if __name__ == "__main__":
    train(saveDir=save,
          numGames=1_000_000,
          opponentSamplingWeight=3,
          trainEvery=25,
          discount=0.8,
          valueScale=10,
          testEvery=300,
          numTestGames=130,
          lr=0.0008,
          wrThresh=0.56,
          boardSize=(6,7),
          examineEvery=25_000,
          showTestGames=False,
          cuda=False)

