#!/usr/bin/env python
# By Nick Erickson
# DDQN Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

# TODO: Prioritized Experience Replay
# TODO: Random agent saving memory/loading memory (Mostly done)
    
from __future__ import print_function

#%%

# Experimental
#import threading

# Utilities
from parameters import hex, gym
import play_game
import agent_ddqn
#import agent_random # Currently broken due to change in memory class

#%%

def main(args):    
    play_game.run(args, agent_ddqn.Agent)
    #play_game.run(args, agent_random.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    args = gym.cart_ddqn
    main(args) 

