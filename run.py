import torch
import torch.nn as nn
import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from c4 import *
import agent
import wandb


def playGame(learner, opponent, boardSize, recordExperience=True, show=False):
    if np.random.uniform(0,1) > 0.5:
        players = [learner, opponent]
        first = True
    else:
        players = [opponent, learner]
        first = False

    board = newBoard(boardSize)
    states, allowed, actions = [], [], []
    for turn in range(board.size+1):
        player = players[turn%2]
        allowT, allow = legalMoves(board)
        if torch.sum(allowT) == 0: val = 0; break
        state = player.observe(board)
        action = player.chooseAction(state, allowT)
        board = player.drop(board, action)
        val = value(board)
        
        #print(action)
        #print(val)
        #print(state.shape)
        if show: printBoard(board)
        
        if recordExperience and (not first)==turn%2:
            states.append(state)
            allowed.append(allow)
            hot = np.eye(player.numActions)[action]
            actions.append(hot)
        if val != 0: break
    if recordExperience: return states, allowed, actions, turn, val
    return turn, val

def train(saveDir,
          numGames=1_000_000,
          opponentSamplingWeight=2,
          trainEvery=35,
          discount=0.98,
          testEvery=300,
          lr=0.01,
          numTestGames=130,
          wrThresh=0.6,
          boardSize=(6,7),
          showTestGames=True,
          cuda=False):


    r = agent.vpoAgent(boardSize, lr=lr, stationary=True, color=1, cuda=cuda)
    r.save(saveDir, "0") # save a random policy to be the first opponent

    learner = agent.vpoAgent(boardSize, lr=lr, stationary=True, color=1, cuda=cuda) # initialize the first agent to be trained
    opponent = agent.vpoAgent(boardSize, lr=lr, stationary=True, color=-1, cuda=cuda) # initialize the first opponent to be trained against

    config = {"numGames":numGames, "opponentSamplingWeight":opponentSamplingWeight, "trainEvery":trainEvery, "discount":discount, "testEvery":testEvery, "lr":lr, "numTestGames":numTestGames, "wrThresh":wrThresh, "boardSize":boardSize, "showTestGames":showTestGames, "optimizer":f"{learner.policy.opt}"}
    wandb.init(project="vpoc4", config=config)
    wandb.watch(learner.policy, log="all", log_freq=10)

    beatBest, wr, loss = False, 0, 0
    opponents = [r.stateDict()] # we keep the previous best models' state_dicts
    for game in (t:=trange(numGames, ncols=130, unit="games")):
        desc = blue + bold
        opIdx = sampleOpponents(len(opponents), weight=opponentSamplingWeight)
        opponent.loadDict(opponents[opIdx])
        desc += f"VSing agent#{opIdx}/{len(opponents)-1}"
        
        states, allowed, actions, numTurns, val = playGame(learner, opponent, boardSize)

        nt = len(actions)
        weights = [100*val*discount**(nt-i) for i in range(nt)] # discounted weights for each state based on game outcome
        learner.remember(states, allowed, actions, weights)

        if game > 0 and game%trainEvery == 0:
            loss = learner.train()
            learner.forget()
        desc += f"{orange}, loss:{loss}"

        if game > 0 and game%testEvery == 0:
            score = 0
            opponent.loadDict(opponents[-1])
            for _ in range(numTestGames):
                numTurns, val = playGame(learner, opponent, boardSize, recordExperience=False, show=showTestGames)
                score += val
            wr = (score/numTestGames + 1)/2
            beatBest = wr >= wrThresh
            if beatBest:
                opponents.append(learner.stateDict())
                learner.save(saveDir, f"{game}")
        if beatBest: desc += f"{green}, beat with wr={wr:.4f}"
        else: desc += f"{red}, lost with wr={wr:.4f}"

        if game > 0 and game%10_000 == 0:
            scores = [0 for i in range(len(opponents))]
            for op in range(len(opponents)):
                opponent.loadDict(opponents[op])
                for _ in range(100):
                    numTurns, val = playGame(learner, opponent, boardSize, recordExperience=False, show=showTestGames)
                    scores[op] += val
            plt.plot(scores)
            plt.show()


        wandb.log({"gameLength": numTurns, "loss":loss, "winrate":wr, "numAgents":len(opponents)})
        t.set_description(desc + purple)

save = "D:\\deepc4\\vpo"

if __name__ == "__main__":
    #play()
    train(saveDir=save,
          numGames=1_000_201,
          opponentSamplingWeight=3,
          trainEvery=25,
          discount=0.98,
          testEvery=300,
          numTestGames=100,
          lr=0.01,
          wrThresh=0.57,
          boardSize=(6,7),
          showTestGames=False,
          cuda=False)

