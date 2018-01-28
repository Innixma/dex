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


# Experimental
# import threading

class Gather:
    def __init__(self, levels, gather_delay, args):
        self.args = copy.deepcopy(args)
        self.env = EnvironmentRealtimeA3C(args.env)
        self.action_dim = self.env.env.action_dim()
        self.state_dim = list(self.env.env.state_dim()) + [args.hyper.img_channels]
        self.levels = levels
        self.gather_delay = gather_delay
        self.levelCount = len(self.levels)
        self.curlevel = 0

    def gather(self):
        for i in range(self.levelCount):
            level_name = self.levels[i]

            print('gathering from ' + level_name)

            delay = self.gather_delay[i]
            args = copy.deepcopy(self.args)
            args.memory_delay = delay
            args.directory = 'gather_' + level_name

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

            print('switching levels')
            # Switch to next level
            self.env.env.env.press('right_arrow')
            time.sleep(0.1)
            self.env.env.env.release('right_arrow')
            time.sleep(0.1)

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

    args = copy.deepcopy(hex.gather_a3c)
    args.hyper.extra.brain_memory_size = 50000

    gather = Gather(levels, gather_delay, args)

    gather.gather()
