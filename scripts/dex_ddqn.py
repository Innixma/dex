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
from param_const import gym_cart_ddqn, hex_base, hex_base_gather, gym_pong_ddqn
import play_game
import agent_ddqn
#import agent_random # Currently broken due to change in memory class

#%%

def main(args):    
    play_game.run(args, agent_ddqn.Agent)
    #play_game.run(args, agent_random.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    #args = hex_base
    #args = hex_base_gather
    #args = gym_cart_ddqn
    args = gym_pong_ddqn
    main(args) 

