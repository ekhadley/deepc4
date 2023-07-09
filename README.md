# Deep learning of Connect4 through self play
This is my implementation of deep reinforcement learning through self play, applied to the game of Connect4. My goal (before starting) is to train an agent which can reliably beat or draw me, somebody who has probably the average amount of Connect4 experience.

The neural networks used by the agent will take in a game state, and output an estimate of its proabibility of winning, for
each of the possible actions.