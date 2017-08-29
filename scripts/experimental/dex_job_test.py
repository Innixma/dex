#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

from __future__ import print_function

import copy

from parameters import hex
from experimental.dex_job_scheduler import GameController, JobData, GameScheduler

# Experimental
# import threading


levelsOrig = [
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

args = copy.deepcopy(hex.base_a3c)
args.hyper.extra.brain_memory_size = 50000

controller = GameController(args, levelsOrig)

levels = [
             'rotation_1',
              'rotation_2',
              'rotation_3',
              'rotation_4',
              'rotation_5',
              'rotation_6',
              'rotation_7'
             ]

delay = [
         4,
         0.5,
         0.5,
         0.5,
         0.5,
         0.5,
         0.5
         ]

name = 'incremental_v1'

full_schedule = JobData(levels, args, delay, name)

phases = full_schedule.phase

# Temporary
phases[0] = []
phases[1] = phases[1][6:]


break_time = 3600

scheduler = GameScheduler(controller, phases, break_time)
scheduler.run()







