#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03
    
from __future__ import print_function

#%%

# Experimental
#import threading

# Utilities
#from parameters.param_const import hex_thinkfast_a3c, hex_gather_a3c, hex_base_a3c, gym_pong_a3c, gym_cart_a3c, hex_base_a3c_load, hex_incongruence_a3c_load, hex_pi_acer_load
from parameters import hex, gym
import play_game
import agent_a3c

#%%



def main(args):
    play_game.run(args, agent_a3c.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    
    levels = [
              'base',
              'base_hard',
              'rotation',
              'hexagon',
              'hexagon_rotation',
              'hexagon_extreme',
              'hexagon_real',
              'think_fast',
              'rotation_fast']
    
    curlevel = 0
    level_idx = 0
    
