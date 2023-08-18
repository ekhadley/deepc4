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
torch.set_printoptions(threshold=150)

def playAgentGame(learner:vpoAgent, opponent:vpoAgent, boardShape, learnerFirst=None, show=False, recordExperience=True):
    learnerFirst = learnerFirst if learnerFirst is not None else np.random.uniform(0, 1) >= 0.5
    players = [opponent, learner]
    pnames = ["opponent", "learner"]
    board = newBoard(boardShape, device=learner.device)
    boardsize = boardShape[0]*boardShape[1]
    if recordExperience:
        states = torch.zeros((boardsize//2 + 1, 2, *boardShape), device=learner.device)
        actions = torch.zeros((boardsize//2 + 1, boardShape[1]), device=learner.device)
    
    lturn, val = 0, 0
    for turn in range(boardsize):
        if show: printBoard(board)
        pid = not learnerFirst == turn%2
        player = players[pid]
        state = player.observe(board)
        dist, action = player.chooseAction(state)
        if recordExperience and pid:
            states[lturn] += state[0]
            actions[lturn] += torch.eye(boardShape[1], device=learner.device)[action]
            lturn += 1
        board = player.drop(board, action)
        if show: print(f"{lemon}\nturn {turn}, {orange}{pnames[pid]}, {pink}{action=}, {lime}val={learner.valnet(board).detach().item():.5f}, {cyan}value={value(board).item()}\n {gray}probs:{dist.detach().numpy().flatten()}", endc)
        if sum(legalActions(board)) == 0: break; print("drawn by no legal actions")
        val = value(board)
        if val != 0: break; print(f"{pnames[pid]} wins!")
    if show: printBoard(board)
    # careful with value conventions. 1 is a win for player 0, -1 is a win for player 1
    # the learning agent is assigned player 0 in the training function. The opponent is player 1.
    if recordExperience: return states[:lturn], actions[:lturn], lturn, val
    return lturn, val

def playWithUser(a, seed=None):
    if isinstance(a, str):
        agent = vpoAgent((6,7), lr=0.001, stationary=True, color=1, cuda=False)
        agent.load(a)
    else: agent = a
    boardsize = agent.boardShape[0]*agent.boardShape[1]

    r = np.random.uniform(0, 1) if seed is None else seed
    userFirst = (r >= 0.5)

    board = newBoard(agent.boardShape)
    printBoard(board)
    for turn in range(boardsize):
        allow = legalActions(board)
        if sum(allow) == 0:
            print(bold, gray, "the game is a draw!", endc)
            return 0
        if (not userFirst) == turn%2:
            dist, action = agent.chooseAction(agent.observe(board))
            print(f"{lemon}\nturn {turn} {pink}{action=}, {lime}val={agent.valnet(board).detach().item():.5f}, {cyan}value={value(board).item()}\n {gray}probs:{dist.detach().numpy().flatten()}", endc)
            board = agent.drop(board, action)
            val = value(board)
            if val != 0:
                print(bold, red, "the agent wins!", endc)
                printBoard(board)
                return val
        else:
            msg = f"{blue}enter move {np.where(allow.detach().numpy())}:\n{endc}"
            validinp = False
            fails = 0
            while not validinp:
                try:
                    inp = input(msg)
                    board = drop(board, int(inp), -1)
                    validinp = True
                except:
                    fails += 1
                    print(red, f"{fails}/10 invalid inputs receieved: {inp}(type:{type(inp)})", endc)
                    assert fails < 11, "too many invalid inputs, erroring out"
            val = value(board)
            if val != 0:
                print(bold, green, "the user wins!", endc)
                printBoard(board)
                return val
        printBoard(board)
        print()
    
    if val != 0:
        print(bold, green, "the user wins!", endc)
        printBoard(board)
        return val

    return 255

def train(saveDir,
          loadDir=None,
          numGames=10_000_000,
          rollback=0,
          opponentSamplingWeight=2,
          adam=False,
          batchSize=35,
          weightDecay=0.003,
          discount=0.8,
          valueScale=1,
          testEvery=300,
          examineEvery=10_000,
          vlr=0.001,
          plr=0.001,
          numTestGames=130,
          wrThresh=0.6,
          boardShape=(6,7),
          showTestGames=True,
          cuda=False):
    boardSize = boardShape[0]*boardShape[1]

    if loadDir is None:
        r = vpoAgent(boardShape, 1, cuda=cuda)
        r.save(saveDir, "0") # save a random policy to be the first opponent
        learner = vpoAgent(boardShape, 0, vlr=vlr, plr=plr, cuda=cuda, wd=weightDecay, adam=adam) # initialize the first agent to be trained
        opponent = vpoAgent(boardShape, 1, cuda=cuda) # initialize the first opponent to be trained against
        opponents = [r.policyStateDict()] # we keep the previous best models' state_dicts here
        startPoint = 0
    elif isinstance(loadDir, str):
        opponent = vpoAgent(boardShape, 1)
        opponents = loadAllModels(loadDir)[0:-rollback-1]
        learner = vpoAgent(boardShape, 0, plr=plr, vlr=vlr, cuda=cuda, wd=weightDecay, adam=adam)
        learner.loadPolicy(opponents[-1])
        startPoint = sorted([int(x.replace(".pth", "")) for x in os.listdir(saveDir)])[-rollback-1]

    config = {"policyModelShape": learner.policy, "valueNetModelShape": learner.valnet, "numGames":numGames, "opponentSamplingWeight":opponentSamplingWeight, "batchSize":batchSize, "discount":discount, "testEvery":testEvery, "policyLR":plr,"valueLR":vlr, "numTestGames":numTestGames, "wrThresh":wrThresh, "boardShape":boardShape, "showTestGames":showTestGames, "optimizer":learner.policy.opt.state_dict(), "leaky_acts":learner.policy.leaky, "valueScale":valueScale}
    wandb.init(project="vpoc4", config=config, dir="D:\\wgmn\\")
    wandb.watch((learner.policy, learner.valnet), log="all", log_freq=10)

    beatBest, matchWR, loss = False, 0, 0
    for game in (t:=trange(numGames, ncols=140, unit="games")):
        if game == 0: game = startPoint
        desc = blue + bold
        
        opIdx = sampleOpponents(len(opponents), weight=opponentSamplingWeight) #sample random opponent 
        opponent.loadPolicy(opponents[opIdx])
        desc += f"VSing agent#{opIdx}/{len(opponents)-1}"
        
        states, actions, numTurns, val = playAgentGame(learner, opponent, boardShape) # play a game vs it
        #weights = rtg(val, numTurns, discount, valueScale)
        learner.addGame(states, actions, val, numTurns) # remember the game/rewards

        if game > 0 and game%batchSize == 0: loss = learner.train()
        desc += f"{orange}, loss:{loss:.4f}"

        if game > 0 and game%testEvery == 0: # play test match to see if learner has surpassed prev best policy
            score = 0
            opponent.loadPolicy(opponents[-1])
            for midx in (mrange:=trange(numTestGames, ncols=80, unit="games", ascii=" >=")):
                m_states, m_actions, m_numTurns, m_val = playAgentGame(learner, opponent, boardShape, learnerFirst=(midx>=numTestGames//2), show=showTestGames) # play a game vs it
                #desc += f"{gray}{dist.detach().numpy()[0]}{endc}"
                #m_weights = rtg(m_val, m_numTurns, discount, valueScale)
                learner.addGame(m_states, m_actions, m_val, m_numTurns) # remember the game/rewards
                if midx > 0 and midx%batchSize == 0: mloss = learner.train()

                score += m_val
                matchWR = (score.detach().item()/(midx+1) + 1)/2
                beatBest = matchWR >= wrThresh
                mrange.set_description((green if beatBest else red) + f"{matchWR:.3f} vs #{len(opponents)-1}")
            if beatBest:
                opponents.append(learner.policyStateDict())
                learner.save(saveDir, f"{game}")
        if beatBest: desc += f"{green}, beat with wr={matchWR:.4f}"
        else: desc += f"{red}, lost with wr={matchWR:.4f}"

        if game > 0 and game%examineEvery == 0: # we pause training every so often to examine the learner's performance
            scores = [0]*len(opponents)
            for op in trange(len(opponents), desc=lemon, ncols=80, unit="ops", ascii=" >="): # if each model is a bit better than the last, we should see a trend of improving match scores as we vs earlier agents
                opponent.loadPolicy(opponents[op])
                for _ in range(numTestGames//2):
                    numTurnssss, valll = playAgentGame(learner, opponent, boardShape, show=showTestGames, recordExperience=False)
                    scores[op] += valll
            playWithUser(learner) # play a game vs the user to test it out
            plt.plot(scores) # plot scores vs all prev agents (we should see correlation)
            plt.show()

        wandb.log({"gameLength": numTurns, "loss":loss, "matchWR":matchWR, "numAgents":len(opponents), "score":val})
        t.set_description(desc + purple)

save = "D:\\wgmn\\deepc4\\ac"

#import cProfile
#prof = cProfile.Profile()
#prof.enable()


if __name__ == "__main__":
    train(saveDir=save,
          loadDir=None,
          rollback=0,
          numGames=100_000_001,
          opponentSamplingWeight=3,
          batchSize=100,
          discount=0.8,
          adam=False,
          weightDecay=0.0005,
          valueScale=1,
          testEvery=500,
          numTestGames=200,
          plr=0.05,
          vlr=0.5,
          wrThresh=0.56,
          boardShape=(6,7),
          examineEvery=300_000,
          showTestGames=False,
          cuda=False)

#prof.dump_stats("tmp")















