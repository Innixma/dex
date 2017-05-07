#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03
    
from __future__ import print_function

# Experimental
#import threading

from parameters import hex
from environment import Environment_realtime_a3c
from play_game import playGameReal_a3c_incremental_init, playGameReal_a3c_incremental
import agent_a3c
import time
import numpy as np
import copy
from data_utils import save_weights

class Incremental:
    def __init__(self, levels, break_time, args):
        self.env = Environment_realtime_a3c(args.env)
        self.action_dim = self.env.env.action_dim()
        self.state_dim = list(self.env.env.state_dim()) + [args.hyper.img_channels]
        self.levels = levels
        self.levelCount = len(self.levels)
        self.break_time = break_time
        self.curlevel = 0
        
    def incremental_learn(self, args, levels_list):
        args = copy.deepcopy(args)
        agent, hasSavedMemory, max_frame_saved = playGameReal_a3c_incremental_init(args, agent_a3c.Agent, self.state_dim, self.action_dim)
        length = len(incremental_levels1)
        idxs = []
        idx_diff = []
        for i in range(length):
            idxs.append(self.levels.index(levels_list[i]))
        idx_diff.append((idxs[0] - self.curlevel) % self.levelCount)
        for i in range(1, length):
            idx_diff.append((idxs[i] - idxs[i-1]) % self.levelCount)
        
        print(idxs)
        print(idx_diff)
        
        print('Switching Levels Initial')
        for i in range(idx_diff[0]):
            self.env.env.env.press('right_arrow')
            time.sleep(0.1)
            self.env.env.env.release('right_arrow')
            time.sleep(0.1)
        self.curlevel = idxs[0]
        
        for i in range(length):
            time_start = time.time()
            while True:
                hasSavedMemory, max_frame_saved = playGameReal_a3c_incremental(agent, self.env, self.state_dim, self.action_dim, hasSavedMemory, max_frame_saved)
                if time.time() - time_start > break_time:
                    break
            if i != length - 1:
                print('Switching Levels')
                for j in range(idx_diff[i+1]):
                    self.env.env.env.press('right_arrow')
                    time.sleep(0.1)
                    self.env.env.env.release('right_arrow')
                    time.sleep(0.1)
                self.curlevel = idxs[i+1]
            
            save_weights(agent, 'id_' + str(i))
            
        agent.brain.init_vars() # Reset network    
    

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
              'rotation_fast'
              ]
    
    incremental_levels1 = ['base', 'base_hard']
    incremental_levels2 = ['base_hard', 'base_hard']
    incremental_levels3 = ['hexagon_rotation', 'hexagon_real']
    incremental_levels4 = ['hexagon_real', 'hexagon_real']
    break_time = 3600   
    curlevel = 0
    level_idx = 0

    incremental = Incremental(levels, break_time, hex.incongruence_a3c)
    
    hex.base_a3c.directory = 'incremental_1'
    incremental.incremental_learn(hex.base_a3c, incremental_levels1)
    
    hex.base_a3c.directory = 'incremental_2'
    incremental.incremental_learn(hex.base_a3c, incremental_levels2)
    
    hex.incongruence_a3c.directory = 'incremental_3'
    incremental.incremental_learn(hex.incongruence_a3c, incremental_levels3)
    
    hex.incongruence_a3c.directory = 'incremental_4'
    incremental.incremental_learn(hex.incongruence_a3c, incremental_levels4)
    