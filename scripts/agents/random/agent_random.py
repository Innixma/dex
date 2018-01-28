# By Nick Erickson
# Random Agent Class

import os
import random

from agents.memory import Memory
from agents.metrics import Metrics
# TODO: Currently broken by new memory class
from utils import globs as G


class Agent:
    def __init__(self, args, state_dim, action_dim):
        self.h = args.hyper
        self.mode = 'observe'
        self.args = args
        self.metrics = Metrics()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory = Memory(self.h.memory_size)
        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1

        if self.args.directory == 'default':
            self.args.directory = G.CUR_FOLDER

        results_location = G.RESULT_FOLDER_FULL + '/' + self.args.directory
        data_location = G.DATA_FOLDER_FULL + '/' + self.args.directory
        os.makedirs(results_location,exist_ok=True)  # Generates results folder
        os.makedirs(data_location,exist_ok=True)  # Generates data folder
        self.results_location = results_location + '/'
        self.data_location = data_location + '/'

    def act(self, s):
        return random.randrange(0, self.action_dim)

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self, debug=False):
        pass
