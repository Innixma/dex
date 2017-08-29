#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

#%%

# Experimental
#import threading

import copy

from agents.a3c import agent_a3c
from environments import play_game
# Utilities
# from parameters.param_const import hex_thinkfast_a3c, hex_gather_a3c, hex_base_a3c, gym_pong_a3c, gym_cart_a3c, hex_base_a3c_load, hex_incongruence_a3c_load, hex_pi_acer_load
from parameters import hex

def main(args):
    play_game.run(args, agent_a3c.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    #args = hex_base_a3c_load

    #args = hex_base_a3c
    #args = gym_cart_a3c
    #args = gym_pong_a3c
    #args = hex_gather_a3c
    #args = hex_thinkfast_a3c
    args = hex.incongruence_a3c
    #args = hex_incongruence_a3c_load
    #args = hex_pi_acer_load

    args = copy.deepcopy(args)
    main(args)

