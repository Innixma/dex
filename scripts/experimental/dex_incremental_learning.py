#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

from __future__ import print_function

import copy
import time

from agents.a3c import agent_a3c
from agents.metrics import Metrics
from environments.environment import EnvironmentRealtimeA3C
from environments.play_game import play_game_real_a3c_incremental_init, play_game_real_a3c_incremental
from parameters import hex
from utils.data_utils import save_weights


# Experimental
# import threading

class Incremental:
    def __init__(self, levels, break_time, args):
        self.env = EnvironmentRealtimeA3C(args.env)
        self.action_dim = self.env.env.action_dim()
        self.state_dim = list(self.env.env.state_dim()) + [args.hyper.img_channels]
        self.levels = levels
        self.levelCount = len(self.levels)
        self.break_time = break_time
        self.curlevel = 0

    def incremental_learn(self, args, levels_list):
        args = copy.deepcopy(args)
        agent, hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental_init(args, agent_a3c.Agent,
                                                                                     self.state_dim, self.action_dim)
        length = len(levels_list)
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
                hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental(agent, self.env, self.state_dim,
                                                                                 self.action_dim, hasSavedMemory,
                                                                                 max_frame_saved)
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

            save_weights(agent, 'id_' + str(i)) # Save weights
            agent.metrics.save(agent.results_location, 'metrics_id_' + str(i)) # Save metrics
            agent.metrics.runs.graph(agent.results_location, 'runs_id_' + str(i))
            agent.metrics = Metrics(agent.metrics.type) # Reset metrics
            agent.brain.metrics = agent.metrics

        agent.brain.init_vars() # Reset network

class Gather_mult:
    def __init__(self, levels, levelpairs, weights, directory, gather_delay, args):
        self.args = copy.deepcopy(args)
        self.levelpairs = levelpairs
        self.weights = weights
        self.directory = directory
        self.env = EnvironmentRealtimeA3C(self.args.env)
        self.action_dim = self.env.env.action_dim()
        self.state_dim = list(self.env.env.state_dim()) + [self.args.hyper.img_channels]
        self.levels = levels
        self.gather_delay = gather_delay
        self.levelCount = len(self.levelpairs)
        self.curlevel = 0

    def gather(self):
        idxs = []
        idx_diff = []
        for i in range(self.levelCount):
            idxs.append(self.levels.index(self.levelpairs[i][1]))
        idx_diff.append((idxs[0] - self.curlevel) % self.levelCount)
        for i in range(1, self.levelCount):
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

        for i in range(self.levelCount):
            #level_name = self.levels[i]
            trained_index = self.levels.index(self.levelpairs[i][0])
            training_index = self.levels.index(self.levelpairs[i][1])
            level_name = self.levels[training_index]
            print('gathering from ' + level_name)

            delay = self.gather_delay[training_index]
            args = copy.deepcopy(self.args)
            args.memory_delay = delay
            args.directory = self.directory[0] #self.directory[trained_index]
            args.weight_override = self.weights[i]

            agent, hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental_init(args, agent_a3c.Agent,
                                                                                         self.state_dim,
                                                                                         self.action_dim)



            while True:
                hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental(agent, self.env, self.state_dim,
                                                                                 self.action_dim, hasSavedMemory,
                                                                                 max_frame_saved)
                if hasSavedMemory:
                    break


            agent.metrics.save(agent.results_location, 'metrics') # Save metrics
            agent.metrics.runs.graph(agent.results_location, 'runs')
            agent.metrics = Metrics(agent.metrics.type) # Reset metrics
            agent.brain.metrics = agent.metrics

            #print('switching levels')
            # Switch to next level
            #$self.env.env.env.press('right_arrow')
            #time.sleep(0.1)
            #self.env.env.env.release('right_arrow')
            #time.sleep(0.1)

            if i != self.levelCount - 1:
                print('Switching Levels')
                for j in range(idx_diff[i+1]):
                    self.env.env.env.press('right_arrow')
                    time.sleep(0.1)
                    self.env.env.env.release('right_arrow')
                    time.sleep(0.1)
                self.curlevel = idxs[i+1]

        print('all done')


if __name__ == "__main__":

    levels = [
              'base_1',
              'base_2',
              'base_3',
              'rotation_1',
              'rotation_2',
              'rotation_3',
              'rotation_4',
              'rotation_5',
              'rotation_6',
              'rotation_7',
              'hexagon_1',
              'hexagon_2',
              'hexagon_3',
              'hexagon_4',
              'thinkfast'
              ]

    #incremental_levels1 = ['base_1', 'base_2', 'base_3']
    #incremental_levels2 = [
    #                      'rotation_1',
    #                      'rotation_2',
    #                      'rotation_3',
    #                      'rotation_4',
    #                      'rotation_5',
    #                      'rotation_6',
    #                      'rotation_7'
    #                      ]
    #incremental_levels3 = ['hexagon_2', 'hexagon_4']

    break_time = 3600
    curlevel = 0
    level_idx = 0

    #break_time = 1200
    incremental = Incremental(levels, break_time, hex.incongruence_a3c)

    hex.base_a3c.hyper.explore = 1
    hex.incongruence_a3c.hyper.explore = 1

    """
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_base_1'
    args.directory = 'base_1'
    incremental.incremental_learn(args, ['base_1'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_base_2'
    args.directory = 'base_2'
    incremental.incremental_learn(args, ['base_2'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_base_3'
    args.directory = 'base_3'
    incremental.incremental_learn(args, ['base_3'])
    
    """
    """
    args = copy.deepcopy(hex.incongruence_a3c)
    args.data = 'gather_rotation_3'
    args.directory = 'rotation_3'
    incremental.incremental_learn(args, ['rotation_3'])
    
    args = copy.deepcopy(hex.incongruence_a3c)
    args.data = 'gather_rotation_5'
    args.directory = 'rotation_5'
    incremental.incremental_learn(args, ['rotation_5'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_hexagon_1'
    args.directory = 'hexagon_1'
    incremental.incremental_learn(args, ['hexagon_1'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_hexagon_2'
    args.directory = 'hexagon_2'
    incremental.incremental_learn(args, ['hexagon_2'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_hexagon_3'
    args.directory = 'hexagon_3'
    incremental.incremental_learn(args, ['hexagon_3'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_hexagon_4'
    args.directory = 'hexagon_4'
    incremental.incremental_learn(args, ['hexagon_4'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_thinkfast'
    args.directory = 'thinkfast'
    incremental.incremental_learn(args, ['thinkfast'])
    
    args = copy.deepcopy(hex.base_a3c)
    args.data = 'gather_rotation_1'
    args.directory = 'rotation_1_v2'
    incremental.incremental_learn(args, ['rotation_1'])
    """

    """
    for i in range(len(levels)):
        level = levels[i]
        
        args = copy.deepcopy(hex.incongruence_a3c)
        args.mode = 'run'
        args.directory = 'test_' + level
        args.weight_override = '../trained_' + level + '/model_max'
        args.hyper.epsilon_init = 0
        args.hyper.epsilon_final = 0
        incremental.incremental_learn(args, [level])
       
    levelpairs = [
                     ['base_1', 'base_2'],
                     ['base_2', 'base_3'],
                     ['base_1', 'rotation_1'],
                     ['rotation_1', 'rotation_2'],
                     ['rotation_2', 'rotation_3'],
                     ['rotation_3', 'rotation_4'],
                     ['rotation_4', 'rotation_5'],
                     ['rotation_5', 'rotation_6'],
                     ['rotation_6', 'rotation_7'],
                     ['hexagon_1', 'hexagon_2'],
                     ['hexagon_2', 'hexagon_3'],
                     ['hexagon_3', 'hexagon_4']
                  ]

    print('moving to next phase...')
    for i in range(len(levels)):
        level1, level2 = levelpairs[i]
        
        args = copy.deepcopy(hex.incongruence_a3c)
        args.mode = 'run'
        args.directory = 'test_' + level1 + '_' + level2
        args.weight_override = '../trained_' + level1 + '/model_max'
        args.hyper.epsilon_init = 0
        args.hyper.epsilon_final = 0
        incremental.incremental_learn(args, [level2])
    """
    """
    levelpairs = [
                 ['base_1', 'base_2'],
                 ['base_2', 'base_3'],
                 ['base_1', 'rotation_1'],
                 ['rotation_1', 'rotation_2'],
                 ['rotation_2', 'rotation_3'],
                 ['rotation_3', 'rotation_4'],
                 ['rotation_4', 'rotation_5'],
                 ['rotation_5', 'rotation_6'],
                 ['rotation_6', 'rotation_7'],
                 ['hexagon_1', 'hexagon_2'],
                 ['hexagon_2', 'hexagon_3'],
                 ['hexagon_3', 'hexagon_4']
              ]
              
    args = copy.deepcopy(hex.gather_a3c)
    args.mode = 'gather'
    args.hyper.extra.brain_memory_size = 40000
    args.hyper.epsilon_init = 1
    args.hyper.epsilon_final = 0.05
    args.hyper.explore = 40000
    gather_delay = [
         4,
         4,
         4,
         4,
         1,
         1,
         1,
         1,
         1,
         1,
         4,
         4,
         4,
         4,
         4
         ]
         
    weights = ['../trained_' + l[0] + '/model_max' for l in levelpairs]
    directory = ['gather_' + l[0] + '_' + l[1] for l in levelpairs]    

    
         
    gather = Gather_mult(levels, levelpairs, weights, directory, gather_delay, args)

    gather.gather()
    """


    gather_delay = [
         4,
         4,
         4,
         4,
         0.5,
         0.5,
         0.5,
         0.5,
         0.5,
         0.5,
         4,
         4,
         4,
         4,
         4
         ]
    levelpairs = [
             #['base_1', 'base_2'],
             #['base_2', 'base_3'],
             #['base_1', 'rotation_1'],
             #['rotation_1', 'rotation_2'],
             #['rotation_2', 'rotation_3'],
             #['rotation_3', 'rotation_4'],
             #['rotation_4', 'rotation_5'],
             #['rotation_5', 'rotation_6'],
             #['rotation_6', 'rotation_7']#,
             #['hexagon_1', 'hexagon_2'],
             #['hexagon_2', 'hexagon_3'],
             #['hexagon_3', 'hexagon_4']
          ]
    levels2 = [
          'rotation_1',
          'rotation_2',
          'rotation_3',
          'rotation_4',
          'rotation_5',
          'rotation_6',
          'rotation_7',
          ]
    """
    for i in range(len(levels2)):
        l1 = levels2[i]
        args = copy.deepcopy(hex.base_a3c)
        args.hyper.extra.brain_memory_size = 40000
        args.hyper.epsilon_init = 0.05
        args.hyper.epsilon_final = 0.05
        args.hyper.explore = 1
        args.memory_delay = gather_delay[levels.index(l1)]
        args.mode = 'train'
        args.data = 'gather_' + l1
        args.directory = 'trained_' + l1
        #args.weight_override = '../trained_' + l1 + '/model_max' 
        incremental.incremental_learn(args, [l1])
    """

    levelpairs = [
     #['base_1', 'base_2'],
     #['base_2', 'base_3'],
     #['base_1', 'rotation_1'],
     #['rotation_1', 'rotation_2'],
     #['rotation_2', 'rotation_3'],
     #['rotation_3', 'rotation_4'],
     #['rotation_4', 'rotation_5'],
     #['rotation_5', 'rotation_6'],
     ['rotation_6', 'rotation_7']#,
     #['hexagon_1', 'hexagon_2'],
     #['hexagon_2', 'hexagon_3'],
     #['hexagon_3', 'hexagon_4']
    ]

    args = copy.deepcopy(hex.gather_a3c)
    args.mode = 'gather'
    args.hyper.extra.brain_memory_size = 100000
    args.hyper.epsilon_init = 0.05
    args.hyper.epsilon_final = 0.05
    args.hyper.explore = 1

    weights = ['../trained_' + l[0] + '_' + l[1] + '/model_max' for l in levelpairs]
    directory = ['experiment_v1/gather_' + l[0] + '_' + l[1] + '_inc' for l in levelpairs]

    gather = Gather_mult(levels, levelpairs, weights, directory, gather_delay, args)

    gather.gather()
    """
    for i in range(len(levelpairs)):
        l1, l2 = levelpairs[i]
        args = copy.deepcopy(hex.base_a3c)
        args.hyper.extra.brain_memory_size = 10000
        args.hyper.epsilon_init = 0.05
        args.hyper.epsilon_final = 0.05
        args.hyper.explore = 1
        args.memory_delay = gather_delay[levels.index(l2)]
        args.mode = 'train_old'
        args.data = 'gather_' + l1 + '_' + l2
        args.directory = 'trained_' + l1 + '_' + l2
        args.weight_override = '../trained_' + l1 + '/model_max' 
        incremental.incremental_learn(args, [l2])
    """

