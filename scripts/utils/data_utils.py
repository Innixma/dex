# By Nick Erickson
# Contains Save/Load Functions

import json
import os
import pickle

import numpy as np

from utils import globs as G


def save_memory_subset(agent, pointer_start, pointer_end, frame_saved, skip=8):
    memory = agent.brain.brain_memory
    if pointer_end < pointer_start:
        pointer_end += memory.max_size

    idxs = []
    for i in range(pointer_start, pointer_end, skip):
        idx = i % memory.max_size
        idxs.append(idx)
    np.savez_compressed(
        agent.data_location+'memory_frame_' + str(frame_saved) + '.npz',
        s  = memory.s [idxs],
        a  = memory.a [idxs],
        r  = memory.r [idxs],
        s_ = memory.s_[idxs],
        t  = memory.t [idxs]
                    )
    print('Memory Saved...')


def load_weights(agent, filename_input=None):  # TODO: Update this function
        if agent.args.directory == 'default':
            agent.args.directory = G.CUR_FOLDER

        results_location = G.RESULT_FOLDER_FULL + '/' + agent.args.directory
        data_location = G.DATA_FOLDER_FULL + '/' + agent.args.directory
        os.makedirs(results_location,exist_ok=True)  # Generates results folder
        os.makedirs(data_location,exist_ok=True)  # Generates data folder
        agent.results_location = results_location + '/'
        agent.data_location = data_location + '/'

        filename = 'model'
        if agent.args.weight_override:
            filename = agent.args.weight_override
        elif agent.args.run_count_load > 0:
            agent.run_count = agent.args.run_count_load
            agent.metrics.total_size = agent.run_count
            filename = filename + '_' + str(agent.args.run_count_load)

        if filename_input:
            filename = filename_input

        if agent.args.mode == 'run':
            try:
                agent.h.extra.observe = 999999999  # Never train
            except:
                pass
            agent.mode = 'observe'
            agent.epsilon = 0
            print("Now we load weight from " + agent.results_location + filename + '.h5')
            agent.brain.model.load_weights(agent.results_location + filename + '.h5')

            print("Weights loaded successfully")
        elif agent.args.mode == 'train_old':  # Continue training old network
            agent.epsilon = agent.h.epsilon_init
            print("Now we load weight from " + agent.results_location + filename + '.h5')
            agent.brain.model.load_weights(agent.results_location + filename + '.h5')
            print("Weights loaded successfully, training")
        elif agent.args.mode == 'gather':  # Gather data, then exit
            print('Gathering Data')
            if agent.args.weight_override:
                agent.epsilon = agent.h.epsilon_init
                print ("Now we load weight from " + agent.results_location + filename + '.h5')
                agent.brain.model.load_weights(agent.results_location + filename + '.h5')
        else:  # Train new
            print('Training new network!')
            agent.epsilon = agent.h.epsilon_init


def save_weights(agent, addon=None):
    name = 'model'
    if addon:
        name = name + '_' + str(addon)
    print("Saving Model...")
    agent.brain.model.save_weights(agent.results_location + name + '.h5', overwrite=True)
    with open(agent.results_location + name + '.json', "w") as outfile:
        json.dump(agent.brain.model.to_json(), outfile)


# Saves memory, hyperparams, and screen info
def save_all(agent):
    save_memory_v2(agent)
    hyper_file = agent.data_location + 'hyper'
    screen_file = agent.data_location + 'screen'
    save_class(agent.h, hyper_file)
    save_class(agent.args.screen, screen_file)


# For Memory_v2 agents
def save_memory_v2(agent):
    memory = agent.brain.brain_memory
    np.savez_compressed(
                        agent.data_location+'memory.npz',
                        s  = memory.s ,
                        a  = memory.a ,
                        r  = memory.r ,
                        s_ = memory.s_,
                        t  = memory.t
                        )
    print('Memory Saved...')

"""
# Saves memory, hyperparameters, and screen parameters
def saveMemory(agent):
    
    d = np.array(agent.memory.D)
    dLen = d.shape[0]

    statesShape  = list(d[0][0].shape)
    states_Shape = list(d[0][3].shape)

    states  = np.zeros([dLen] + statesShape, dtype='float16')
    actions = np.zeros(dLen, dtype='int_')
    rewards = np.zeros(dLen, dtype='float64')
    states_ = np.zeros([dLen] + states_Shape, dtype='float16')
    terminals = np.zeros(dLen, dtype='bool_')
    
    for i in range(dLen):
        states[i]    = d[i][0]
        actions[i]   = d[i][1]
        rewards[i]   = d[i][2]
        states_[i]   = d[i][3]
        terminals[i] = d[i][4]
    
        #print(np.mean(states[i]))

    np.savez_compressed(
                        agent.data_location+'memory.npz',
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        states_=states_,
                        terminals=terminals
                        )
    #a = np.load(agent.results_location+'memory.npz')
    #return #a
"""


# Saves class info
def save_class(object_, location):
    with open(location,"wb") as file:
        pickle.dump(object_, file)


def load_class(location):
    with open(location,"rb") as file:
        return pickle.load(file)


def load_memory_v2(agent, memory_location, extra=''):
    memory = np.load(memory_location + 'memory' + extra + '.npz')
    agent.brain.brain_memory.s  = memory['s' ]
    agent.brain.brain_memory.a  = memory['a' ]
    agent.brain.brain_memory.r  = memory['r' ]
    agent.brain.brain_memory.s_ = memory['s_']
    agent.brain.brain_memory.t  = memory['t' ]

    agent.brain.brain_memory.size = agent.brain.brain_memory.s.shape[0]
    agent.brain.brain_memory.max_size = agent.brain.brain_memory.size
    agent.brain.brain_memory.total_saved = agent.brain.brain_memory.size
    agent.brain.brain_memory.is_full = True
    print('Importing', agent.brain.brain_memory.size, 'states')


def load_memory_direct(memory_location, extra=''):
    memory = np.load(memory_location + 'memory' + extra + '.npz')
    s  = memory['s' ]
    a  = memory['a' ]
    r  = memory['r' ]
    s_ = memory['s_']
    t  = memory['t' ]

    return s, a, r, s_, t

"""
def loadMemory(agent, memory_location):
    memory = np.load(memory_location+'memory.npz')
    
    states    = memory['states']
    actions   = memory['actions']
    rewards   = memory['rewards']
    states_   = memory['states_']
    terminals = memory['terminals']

    memoryLen = states.shape[0]

    print('Importing', memoryLen, 'states:')
    for i in range(memoryLen):
        if i % 10000 == 0:
            print(i,'/',memoryLen)
        agent.memory.add([states[i], actions[i], rewards[i], states_[i], terminals[i]])
    print('Import complete!')
"""
