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
from metrics import Metrics
from data_utils import save_weights, loadMemory_direct
import os
from dex_job_scheduler import GameController, JobData, GameScheduler
import globs
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_filters(weights, x, y):
    
    filters = weights
    numFilters = filters.shape[-1]
    fig = plt.figure()
    print(filters.shape)
    for j in range(numFilters):
        ax = fig.add_subplot(y, x, j+1)
        ax.matshow(filters[:,:,0,j], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
    
levels = [
             'rotation_1',
              'rotation_2',
              'rotation_3',
              'rotation_4',
              'rotation_5',
              'rotation_6',
              'rotation_7'
             ]
             
name = 'trained_rotation_1'
         
directory = globs.RESULT_FOLDER_FULL + '/'

weights_name = 'model_frame_16291'

weights_dir = directory + name + '/' + weights_name

weights_dir_h5 = weights_dir + '.h5'

weights_dir_json = weights_dir + '.json'


overrides = ['../trained_rotation_1/model_frame_419','../trained_rotation_1/model_frame_16291']

memory_location = '../data/' + 'trained_rotation_1' + '/'
s, a, r, s_, t = loadMemory_direct(memory_location)

image = s[1995:1997]

#args.weight_override = '../trained_rotation_1/model_frame_16291'
for j in overrides:
    args = copy.deepcopy(hex.base_a3c)
    args.weight_override = j
    args.dir = 'testing_vis'
    args.mode = 'run'
    state_dim = [42, 42, 2]
    action_dim = 3
    agent, _, _ = playGameReal_a3c_incremental_init(args, agent_a3c.Agent, state_dim, action_dim)
    model = agent.brain.model
    config = model.get_config()
    count = model.count_params()
    numLayers = len(model.layers)
    weights_full = model.get_weights()
    weights = weights_full[0]
    layer = model.layers[0]
    plot_filters(weights, 8, 8)
plt.show()

sess = agent.brain.session
graph = agent.brain.graph
s_t, a_t, r_t, minimize, loss_total, log_prob, loss_policy, loss_value, entropy = graph



z = sess.run(image)

#output_layer = model.layers[0].get_output()
#output_fn = tf.function



















