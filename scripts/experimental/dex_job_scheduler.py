#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

from __future__ import print_function

import copy
import shutil
import time
from os import listdir
from os.path import isfile, join

import numpy as np

from agents.a3c import agent_a3c
from agents.metrics import Metrics
from environments.environment import EnvironmentRealtimeA3C
from environments.play_game import play_game_real_a3c_incremental_init, play_game_real_a3c_incremental
from utils.data_utils import save_weights


# Experimental
# import threading

class GameController:
    def __init__(self, args, levels):
        self.args = copy.deepcopy(args)
        self.levels = copy.deepcopy(levels)
        self.size = len(self.levels)
        self.idx = 0
        self.env = EnvironmentRealtimeA3C(self.args.env)

    def switch(self, level):
        if isinstance(level, int):
            idx_new = level
        else:
            idx_new = self.levels.index(level)

        idx_diff = (idx_new - self.idx) % self.size
        #print(idx_new, idx_diff)
        for i in range(idx_diff):
            self.env.env.env.press('right_arrow')
            time.sleep(0.1)
            self.env.env.env.release('right_arrow')
            time.sleep(0.1)

        self.idx = idx_new

    def reset(self):
        self.switch(0)

class JobData:
    def __init__(self, levels, args, delay, name):
        self.levels = levels
        self.args = copy.deepcopy(args)
        self.delay = delay
        self.name = name
        self.size = len(self.levels)

        self.phase = []
        self.prefixes = [self.name + '/phase_' + str(phase) for phase in range(self.size)]
        for i in range(self.size):
            jobs = self.generate_jobs(i)
            self.phase.append(jobs)

    def generate_jobs_init(self):
        pass

    def generate_jobs(self, phase):
        #if phase == 0:
            #return self.generate_jobs_init()

        jobs = []
        jobs.extend(self.generate_gather(phase))
        jobs.extend(self.generate_train(phase))
        return jobs

    def generate_gather(self, phase):
        prefix = self.prefixes[phase]
        prefix += '/gather/'

        levels = self.levels[phase:]

        dirs = []
        weights = []
        data = []
        mode = []
        delay = []

        for i in range(len(levels)):
            level = levels[i]
            idx = self.levels.index(level)
            prevLevel = self.levels[self.levels.index(level)-1]
            dirs.append(prefix + level)
            if phase == 0:
                weights.append(None)
            else:
                weights.append('../../../../' + self.prefixes[phase-1] + '/trained/' + prevLevel + '/model_max')
            data.append(None)
            mode.append('gather')
            delay.append(self.delay[idx])

        jobs = []
        for i in range(len(levels)):
            job = Job(levels[i], dirs[i], weights[i], data[i], mode[i], self.args, delay[i])
            jobs.append(job)

        return jobs

    def generate_train(self, phase):
        prefix = self.prefixes[phase]
        prefix += '/trained/'

        levels = self.levels[phase:]

        dirs = []
        weights = []
        data = []
        mode = []
        delay = []

        for i in range(len(levels)):
            level = levels[i]
            idx = self.levels.index(level)
            prevLevel = self.levels[self.levels.index(level)-1]
            dirs.append(prefix + level)
            if phase == 0:
                weights.append(None)
            else:
                weights.append('../../../../' + self.prefixes[phase-1] + '/trained/' + prevLevel + '/model_max')
            data.append(self.prefixes[phase] + '/gather/' + level)
            if phase == 0:
                mode.append('train')
            else:
                mode.append('train_old')
            delay.append(self.delay[idx])

        jobs = []
        for i in range(len(levels)):
            job = Job(levels[i], dirs[i], weights[i], data[i], mode[i], self.args, delay[i])
            jobs.append(job)

        return jobs


class Job:
    def __init__(self, level, directory, weights, data, mode, args, delay):

        self.level = level
        self.dir = directory
        self.weights = weights
        self.data = data
        self.mode = mode
        self.delay = delay
        self.args = copy.deepcopy(args)

        self.args.directory = self.dir
        self.args.data = data
        self.args.weight_override = self.weights
        self.args.mode = self.mode
        self.args.memory_delay = self.delay
        if weights == None and mode == 'gather':
            self.args.hyper.epsilon_init = 1
            self.args.hyper.epsilon_final = 1
            self.args.hyper.explore = 9999999
        else:
            self.args.hyper.epsilon_init = 0.05
            self.args.hyper.epsilon_final = 0.05
            self.args.hyper.explore = 1

class GameScheduler:
    def __init__(self, controller, phases, break_time):
        self.controller = controller
        self.env = self.controller.env
        self.phases = phases
        self.state_dim = list(self.env.env.state_dim()) + [self.controller.args.hyper.img_channels]
        self.action_dim = self.env.env.action_dim()
        self.break_time = break_time

    def run(self):
        for phase in self.phases:
            for job in phase:
                self.run_job(job)

    def run_job(self, job):
        args = job.args
        level = job.level

        self.controller.switch(level)

        agent, hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental_init(args, agent_a3c.Agent,
                                                                                     self.state_dim, self.action_dim)


        time_start = time.time()
        while True:
            hasSavedMemory, max_frame_saved = play_game_real_a3c_incremental(agent, self.env, self.state_dim,
                                                                             self.action_dim, hasSavedMemory,
                                                                             max_frame_saved)
            if agent.args.mode == 'gather':
                if hasSavedMemory:
                    break
            elif time.time() - time_start > self.break_time:
                break

        save_weights(agent, 'end') # Save weights
        agent.metrics.save(agent.results_location, 'metrics_end') # Save metrics
        agent.metrics.runs.graph(agent.results_location, 'runs_end')
        agent.metrics = Metrics(agent.metrics.type) # Reset metrics
        agent.brain.metrics = agent.metrics
        agent.brain.init_vars() # Reset network

        if agent.args.mode != 'gather':
            directory = agent.results_location
            allfiles = listdir(directory)
            onlyfiles = [f for f in allfiles if isfile(join(directory, f))]
            frame = [f for f in onlyfiles if 'model_frame' in f and '.h5' in f]
            frame_time = [int(f[12:-3]) for f in frame]
            if frame_time == []:
                max_file_name = 'model_end.h5'
            else:
                max_file_name = frame[np.argmax(frame_time)]

            max_file = join(directory,max_file_name)

            src = max_file
            dst = join(directory, 'model_max.h5')
            shutil.copyfile(src, dst)

