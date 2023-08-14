import torch
import torch.nn as nn
import numpy as np
import os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from c4 import *
from agent import vpoAgent
import wandb
np.set_printoptions(suppress=True, linewidth=200, precision=4)

def playAgentGame(learner:vpoAgent, opponent:vpoAgent, boardShape, wMask, recordExperience=True, show=False, seed=None):
    r = np.random.uniform(0, 1) if seed is None else seed
    if r >= 0.5:
        players = [learner, opponent]
        learnerFirst = True
    else:
        players = [opponent, learner]
        learnerFirst = False

    board = newBoard(boardShape)
    states = []
    dists = torch.zeros((board.size, boardShape[1]), device="cuda" if learner.cuda_ else "cpu")
    actions = torch.zeros((board.size, boardShape[1]), device="cuda" if learner.cuda_ else "cpu")
    for turn in range(board.size):
        player = players[turn%2]
        allow = legalMoves(board)
        allowT = torch.tensor(allow)
        if torch.sum(allowT) == 0: val = 0; break
        state = player.observe(board)
        dist, action = player.chooseAction(state, allowT)
        
        board = player.drop(board, action)
        val = value(board, wMask)
        if show:
            print(gray, dist.detach().numpy()[0], endc)
            printBoard(board)
            print()
        
        if recordExperience and (not learnerFirst)==turn%2:
            dists[len(states)] += dist
            actions[len(states)][action] = 1
            states.append(state)
        if val != 0: break
    numturns = len(states)
    if recordExperience: return states, actions[:numturns], dists[:numturns], numturns, val
    return turn, val

def playWithUser(a, wMask, seed=None):
    if isinstance(agent, str):
        agent = vpoAgent((6,7), lr=0.001, stationary=True, color=1, cuda=False)
        agent.load(a)
    else: agent = a

    r = np.random.uniform(0, 1) if seed is None else seed
    userFirst = (r >= 0.5)

    board = newBoard(agent.boardShape)
    printBoard(board)
    for turn in range(board.size+1):
        allow = legalMoves(board)
        if np.sum(allow) == 0:
            val = 0
            printBoard(board)
            print(gray, "game drawn", endc)
            return val
        if (not userFirst) == turn%2:
            state = agent.observe(board)
            dist, action = agent.chooseAction(state, torch.tensor(allow))
            board = agent.drop(board, action)
            print(f"{gray}{dist.detach().numpy()[0]}{endc}")
            val = value(board, wMask)
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
                    board = drop(board, int(inp), -1)
                    validinp = True
                except:
                    fails += 1
                    print(red, f"invalid input. receieved: {inp}(type:{type(inp)})", endc)
                    assert fails < 11, "too many invalid inputs, erroring out"
            val = value(board, wMask)
            if val != 0:
                print(bold, green, "the user wins!", endc)
                printBoard(board)
                return val

        printBoard(board)
        print()
    return 255

def train(saveDir,
          loadDir=None,
          numGames=10_000_000,
          rollback=0,
          opponentSamplingWeight=2,
          trainEvery=35,
          weightDecay=0.003,
          discount=0.8,
          valueScale=1,
          testEvery=300,
          examineEvery=10_000,
          lr=0.001,
          numTestGames=130,
          wrThresh=0.6,
          boardShape=(6,7),
          showTestGames=True,
          cuda=False):
    wMask = winMask(boardShape)
    boardSize = boardShape[0]*boardShape[1]

    if loadDir is None:
        r = vpoAgent(boardShape, lr=lr, stationary=True, color=1, cuda=cuda, wd=weightDecay)
        r.save(saveDir, "0") # save a random policy to be the first opponent
        learner = vpoAgent(boardShape, lr=lr, stationary=True, color=1, cuda=cuda, wd=weightDecay, memSize=boardSize*trainEvery) # initialize the first agent to be trained
        opponent = vpoAgent(boardShape, lr=lr, stationary=True, color=-1, cuda=cuda, wd=weightDecay) # initialize the first opponent to be trained against
        opponents = [r.stateDict()] # we keep the previous best models' state_dicts here
        startPoint = 0
    elif isinstance(loadDir, str):
        opponent = vpoAgent(boardShape, lr=lr, stationary=True, color=-1, cuda=cuda, wd=weightDecay)
        opponents = loadAllModels(loadDir)[0:-rollback-1]
        learner = vpoAgent(boardShape, lr=lr, stationary=True, color=1, cuda=cuda, wd=weightDecay, memSize=boardSize*trainEvery)
        learner.loadPolicy(opponents[-1])
        startPoint = sorted([int(x.replace(".pth", "")) for x in os.listdir(saveDir)])[-rollback-1]

    config = {"modelShape": learner.policy, "numGames":numGames, "opponentSamplingWeight":opponentSamplingWeight, "trainEvery":trainEvery, "discount":discount, "testEvery":testEvery, "lr":lr, "numTestGames":numTestGames, "wrThresh":wrThresh, "boardShape":boardShape, "showTestGames":showTestGames, "optimizer":learner.policy.opt.state_dict(), "leaky_acts":learner.policy.leaky, "valueScale":valueScale}
    wandb.init(project="vpoc4", config=config, dir="D:\\wgmn\\")
    wandb.watch(learner.policy, log="all", log_freq=10)

    beatBest, matchWR, loss = False, 0, 0
    for game in (t:=trange(numGames, ncols=140, unit="games")):
        if game == 0: game = startPoint
        desc = blue + bold
        opIdx = sampleOpponents(len(opponents), weight=opponentSamplingWeight) #sample random opponent 
        opponent.loadPolicy(opponents[opIdx])
        desc += f"VSing agent#{opIdx}/{len(opponents)-1}"
        
        states, actions, dists, numTurns, val = playAgentGame(learner, opponent, boardShape, wMask, show=False) # play a game vs it
        weights = rtg(val, numTurns, discount, valueScale)
        learner.addEp(dists, actions, weights, numTurns) # remember the game/rewards

        #if game > 0 and game%trainEvery == 0:
        #    with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
        #        loss = learner.train() # train periodically
        #    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cpu_time_total', row_limit=100))
        if game > 0 and game%trainEvery == 0: loss = learner.train()
        
        desc += f"{orange}, loss:{loss:.4f}"

        if game > 0 and game%testEvery == 0: # play test match to see if learner has surpassed prev best policy
            score = 0
            opponent.loadPolicy(opponents[-1])
            for midx in (mrange:=trange(numTestGames, ncols=80, unit="games", ascii=" >=")):
                mstates, mactions, mdists, mnumTurns, mval = playAgentGame(learner, opponent, boardShape, wMask, show=False) # play a game vs it
                #desc += f"{gray}{dist.detach().numpy()[0]}{endc}"
                mweights = rtg(mval, mnumTurns, discount, valueScale)
                learner.addEp(mdists, mactions, mweights, mnumTurns) # remember the game/rewards
                if midx > 0 and midx%trainEvery == 0: mloss = learner.train()

                score += mval
                matchWR = (score/(midx+1) + 1)/2
                beatBest = matchWR >= wrThresh
                mrange.set_description((green if beatBest else red) + f"{matchWR:.3f} vs #{len(opponents)-1}")
            if beatBest:
                opponents.append(learner.stateDict())
                learner.save(saveDir, f"{game}")
        if beatBest: desc += f"{green}, beat with wr={matchWR:.4f}"
        else: desc += f"{red}, lost with wr={matchWR:.4f}"

        if game > 0 and game%examineEvery == 0: # we pause training every so often to examine the learner's performance
            scores = [0 for i in range(len(opponents))]
            for op in trange(len(opponents), desc=yellow, ncols=100, unit="opponents"): # if each model is a bit better than the last, we should see a trend of improving match scores as we vs earlier agents
                opponent.loadPolicy(opponents[op])
                for _ in range(numTestGames):
                    numTurns_, val_ = playAgentGame(learner, opponent, boardShape, wMask, recordExperience=False, show=showTestGames)
                    scores[op] += val_
            playWithUser(learner, wMask) # play a game vs the user to test it out
            plt.plot(scores) # plot scores vs all prev agents (we should see correlation)
            plt.show()

        wandb.log({"gameLength": numTurns, "loss":loss, "matchWR":matchWR, "numAgents":len(opponents), "score":val})
        t.set_description(desc + purple)

save = "D:\\wgmn\\deepc4\\vpo"


#import cProfile
#prof = cProfile.Profile()
#prof.enable()

if __name__ == "__main__":
    train(saveDir=save,
          loadDir=None,
          rollback=0,
          numGames=10_000_000,
          opponentSamplingWeight=3,
          trainEvery=50,
          discount=0.9,
          weightDecay=0.011,
          valueScale=1,
          testEvery=500,
          numTestGames=200,
          lr=0.01,
          wrThresh=0.56,
          boardShape=(6,7),
          examineEvery=100_000,
          showTestGames=False,
          cuda=True)
    
#prof.disable()
#prof.dump_stats("tmp")