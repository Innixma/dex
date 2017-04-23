#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow v12 and Keras v2.02

# Done: Target Network https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
# TODO: Prioritized Experience Replay
# TODO: Random agent saving memory/loading memory (Mostly done)
    
from __future__ import print_function

#%%

# Experimental
#import threading

# Utilities
from param_const import hex_base_a3c, gym_cart_a3c, hex_base_a3c_load, hex_incongruence_a3c, hex_incongruence_a3c_load, hex_pi_acer_load
import play_game
import agent_a3c

#%%

def main(args):    
    play_game.run(args, agent_a3c.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    args = hex_base_a3c_load
    #args = hex_base_a3c
    #args = gym_cart_a3c
    #args = hex_incongruence_a3c
    #args = hex_incongruence_a3c_load
    #args = hex_pi_acer_load
    main(args) 

