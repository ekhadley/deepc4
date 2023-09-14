import torch
import torch.nn as nn
import numpy as np
import re, os, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from c4 import *
from agent import vpoAgent
import wandb
np.set_printoptions(suppress=True, linewidth=200, precision=4)
torch.set_printoptions(threshold=100, sci_mode=False, linewidth=1000, precision=4, edgeitems=4)

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

def playAgentGame(learner:vpoAgent, opponent:vpoAgent, boardShape, learnerFirst=None, show=False, recordExperience=True):
    maxGameLen = boardShape[0]*boardShape[1]
    learnerFirst = learnerFirst if learnerFirst is not None else np.random.uniform(0, 1) >= 0.5
    players = [opponent, learner]
    pnames = ["opponent", "learner"]
    board = newBoard(boardShape, device=learner.device)
    if recordExperience:
        states = torch.zeros((maxGameLen//2 + 1, 2, *boardShape), device=learner.device)
        actions = torch.zeros((maxGameLen//2 + 1, boardShape[1]), device=learner.device)
    
    lturn, val = 0, 0
    for turn in range(maxGameLen):
        if show: printBoard(board)
        pid = not learnerFirst == turn%2
        player = players[pid]
        state = player.observe(board)
        dist, action = player.chooseAction(state)
        if recordExperience and pid:
            states[lturn] += state[0]
            #actions[lturn] += torch.eye(boardShape[1], device=learner.device)[action]
            actions[lturn] += F.one_hot(action, boardShape[1])
            lturn += 1
        board = player.drop(board, action)
        if show: print(f"{lemon}\nturn {turn}, {orange}{pnames[pid]}, {pink}{action=}, {lime}val={learner.valnet(board).detach().item():.5f}, {cyan}value={value(board).item()}\n {gray}probs:{dist.detach().numpy().flatten()}", endc)
        if sum(legalActions(board)) == 0: break; print("drawn by no legal actions")
        val = value(board)
        if val != 0: break; print(f"{pnames[pid]} wins!")
    if show: printBoard(board)
    # careful with value conventions. 1 is a win for player 0, -1 is a win for player 1
    # the learning agent is assigned player 0 in the training function. The opponent is player 1.
    if recordExperience: return states[:lturn], actions[:lturn], val, lturn
    return lturn, val

def playAgentMatch(learner:vpoAgent, opponent:vpoAgent, boardShape, numGames, show=False):
    maxGameLen = boardShape[0]*boardShape[1]
    learnerFirstBoards = newBoard((numGames//2, *boardShape), device=learner.device)

    opponentFirstBoards = newBoard((numGames - numGames//2, *boardShape), device=learner.device)
    dists, acts = opponent.chooseAction(opponentFirstBoards)
    opponentFirstBoards = opponent.drop(opponentFirstBoards, acts)

    boards = torch.cat((learnerFirstBoards, opponentFirstBoards), dim=0)

    recorded_steps, numLiveGames = 0, numGames
    liveGames = torch.arange(0, numGames, device=learner.device, dtype=torch.int32)
    memPositions = [[i] for i in range(numGames)]
    states = torch.zeros((numGames*maxGameLen//2+1, 2, *boardShape), device=learner.device, requires_grad=False)
    actions = torch.zeros((numGames*maxGameLen//2+1, boardShape[1]), device=learner.device, requires_grad=False)
    values = torch.zeros((numGames*maxGameLen//2+1), device=learner.device, requires_grad=False)

    for turn in range(maxGameLen-1):
        learnersMove = not turn%2
        agent = learner if learnersMove else opponent
        dists, acts = agent.chooseAction(agent.observe(boards))
        
        if learnersMove:
            states[recorded_steps:recorded_steps+numLiveGames] = boards
            actions[recorded_steps:recorded_steps+numLiveGames] = F.one_hot(acts, boardShape[1])
        boards = agent.drop(boards, acts)

        vals = value(boards)
        liveidx = torch.where(vals == 0)[0]
        
        if show:
            print(f"{lemon}\nturn {turn}{orange}({'learner' if learnersMove else 'opponent'}), {pink}{acts=}, \n {gray}probs:{dists.detach().numpy()}", endc)
            print(bold, orange, f"{vals=}, {red}{liveidx=}, {red}{liveGames=}, {cyan}{numLiveGames=}\n\n" + endc)
            printBoard(boards)
        
        for i, val in enumerate(vals):
            gameid = liveGames[i]
            if learnersMove:
                memPositions[gameid].append(recorded_steps + i)
                #try: values[recorded_steps + i] += 1 if acts[i]%2 else -1
                #except IndexError: pass
            if val != 0:
                #values[memPositions[gameid]] += 0
                values[memPositions[gameid]] += val/len(memPositions[gameid])
                #values[memPositions[gameid]] += rtg(val, len(memPositions[gameid]), 0.8, 10)/len(memPositions[gameid])
        
        if learnersMove: recorded_steps += numLiveGames
        liveGames = liveGames[liveidx]
        numLiveGames = liveGames.shape[0]
        
        #printValueAttr(values, memPositions, recorded_steps)
        boards = boards[liveidx]
        if numLiveGames == 0: break
    return states[:recorded_steps], actions[:recorded_steps], values[:recorded_steps], recorded_steps

def train(saveDir, loadDir=None, rollback=0,boardShape=(6,7),numGames=100_000_001,opponentSamplingWeight=2,showTestGames=False,valueScale=1,discount=0.8,examineEvery=1_000_000,matchSize=100,testEvery=10,testMatchSize=300,wrThresh=0.566,plr=0.1,vlr=1.0,weightDecay=0.0001,adam=False,cuda=False):
    boardSize = boardShape[0]*boardShape[1]
    numMatches = numGames//matchSize

    if loadDir is None:
        r = vpoAgent(boardShape, 1, cuda=cuda)
        if saveDir is not None: r.save(saveDir, "0") # save a random policy to be the first opponent
        learner = vpoAgent(boardShape, 0, vlr=vlr, plr=plr, cuda=cuda, wd=weightDecay, adam=adam) # initialize the first agent to be trained
        opponent = vpoAgent(boardShape, 1, cuda=cuda) # initialize the first opponent to be trained against
        opponents = [r.policyStateDict()] # we keep the previous best models' state_dicts here
    elif isinstance(loadDir, str):
        opponent = vpoAgent(boardShape, 1)
        policies, valnets = loadAllModels(loadDir)
        opponents = policies[0:-rollback-1]
        learner = vpoAgent(boardShape, 0, plr=plr, vlr=vlr, cuda=cuda, wd=weightDecay, adam=adam)
        learner.loadPolicy(opponents[-1])
        learner.loadValnet(valnets[-rollback-1])

    config = {"policyModelShape": learner.policy, "valueNetModelShape": learner.valnet, "numGames":numGames, "opponentSamplingWeight":opponentSamplingWeight, "matchSize":matchSize, "discount":discount, "testEvery":testEvery, "policyLR":plr,"valueLR":vlr, "numTestGames":testMatchSize, "wrThresh":wrThresh, "boardShape":boardShape, "showTestGames":showTestGames, "optimizer":learner.policy.opt.state_dict(), "leaky_acts":learner.policy.leaky, "valueScale":valueScale}
    wandb.init(project="vpoc4", config=config, dir="D:\\wgmn\\")
    wandb.watch((learner.policy, learner.valnet), log="all", log_freq=10)

    beatBest, matchWR, lastwinWR = False, 0, 0
    for match in (t:=trange(numMatches, ncols=200, unit="matches")):
        desc = blue + bold
        
        opIdx = sampleOpponents(len(opponents), weight=opponentSamplingWeight) #sample random opponent 
        opponent.loadPolicy(opponents[opIdx])
        desc += f"VSing agent#{opIdx}/{len(opponents)-1}"
        
        #states, actions, numTurns, val = playAgentGame(learner, opponent, boardShape) # play a game vs it
        states, actions, val, numTurns = playAgentMatch(learner, opponent, boardShape, matchSize) # play a game vs it
        learner.addGame(states, actions, val, numTurns) # remember the game/rewards

        vloss, ploss, pred_acc = learner.train()
        desc += f"{orange}, vloss:{ploss:.4f}, ploss:{ploss:.4f}"

        if match > 0 and match%testEvery == 0: # play test match to see if learner has surpassed prev best policy
            opponent.loadPolicy(opponents[-1])
            m_states, m_actions, m_vals, m_numTurns = playAgentMatch(learner, opponent, boardShape, testMatchSize, show=showTestGames) # play a game vs it
            learner.addGame(m_states, m_actions, m_vals, m_numTurns) # remember the game/rewards
            learner.train()
            matchWR = (torch.sum(torch.sign(m_vals[:testMatchSize])).item()/(testMatchSize) + 1)/2
            beatBest = matchWR >= wrThresh
            if beatBest:
                opponents.append(learner.policyStateDict())
                lastwinWR = matchWR
                if saveDir is not None: learner.save(saveDir, f"{match}")
        if beatBest: desc += f"{green}, beat with wr={matchWR:.4f}, {lime}last win was with wr={lastwinWR:.4f}"
        else: desc += f"{red}, lost with wr={matchWR:.4f}, {lime}last win was with wr={lastwinWR:.4f}"

        if match > 0 and match%examineEvery == 0: # we pause training every so often to examine the learner's performance
            scores = [0]*len(opponents)
            for op in trange(len(opponents), desc=lemon, ncols=80, unit="ops", ascii=" >="): # if each model is a bit better than the last, we should see a trend of improving match scores as we vs earlier agents
                opponent.loadPolicy(opponents[op])
                __, __, valll, __ = playAgentMatch(learner, opponent, boardShape, matchSize)
                scores[op] += torch.sum(valll).item()
            playWithUser(learner) # play a game vs the user to test it out
            plt.plot(scores) # plot scores vs all prev agents (we should see correlation)
            plt.show()

        wandb.log({"gameLength": numTurns/matchSize, "pred_acc":pred_acc, "vloss":vloss, "ploss":ploss, "matchWR":matchWR, "numAgents":len(opponents), "score":torch.sum(val).item(), "pred_acc":pred_acc}, step=match*matchSize)
        t.set_description(desc + purple)

save = "D:\\wgmn\\deepc4\\ac"

#import cProfile
#prof = cProfile.Profile()
#prof.enable()

if __name__ == "__main__":
    train(saveDir=save,
          loadDir=None,
          rollback=0,
          boardShape=(6,7),
          numGames=1_000_000_000,
          opponentSamplingWeight=2,
          showTestGames=False,
          valueScale=1,
          discount=0.8,
          examineEvery=20_000,
          matchSize=100,
          testEvery=30,
          testMatchSize=300,
          wrThresh=0.566,
          plr=0.2,
          vlr=0.5,
          weightDecay=0.003,
          adam=False,
          cuda=True)

#prof.dump_stats("tmp")