# By Nick Erickson
# Contains Save/Load Functions

import globs as G
import numpy as np
import pickle
import os
import json

def load_weights(agent): # TODO: Update this function
        if agent.args.directory == 'default':
            agent.args.directory = G.CUR_FOLDER

        results_location = G.RESULT_FOLDER_FULL + '/' + agent.args.directory
        data_location = G.DATA_FOLDER_FULL + '/' + agent.args.directory
        os.makedirs(results_location,exist_ok=True) # Generates results folder
        os.makedirs(data_location,exist_ok=True) # Generates data folder
        agent.results_location = results_location + '/'
        agent.data_location = data_location + '/'
        
        filename = 'model'
        if agent.args.run_count_load > 0:
            agent.run_count = agent.args.run_count_load
            agent.metrics.total_size = agent.run_count
            filename = filename + '_' + str(agent.args.run_count_load)
        
        if agent.args.mode == 'run':
            agent.h.observe = 999999999    # Never train
            agent.epsilon = 0
            print ("Now we load weight from " + agent.results_location + filename + '.h5')
            agent.brain.model.load_weights(agent.results_location + filename + '.h5')

            print ("Weights loaded successfully")
        elif agent.args.mode == 'train_old': # Continue training old network
            agent.h.observe = agent.h.observe
            agent.epsilon = agent.h.epsilon_init
            print ("Now we load weight from " + agent.results_location + filename + '.h5')
            agent.brain.model.load_weights(agent.results_location + filename + '.h5')

            print ("Weights loaded successfully, training")
        else: # Train new
            print('Training new network!')
            agent.h.observe = agent.h.observe
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
def saveAll(agent):
    saveMemory(agent)
    hyperfile = agent.data_location + 'hyper'
    screenfile = agent.data_location + 'screen'
    saveClass(agent.h, hyperfile)
    saveClass(agent.args.screen, screenfile)
            
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


    
# Saves class info
def saveClass(object_, location):
    with open(location,"wb") as file:
        pickle.dump(object_, file)
    
def loadClass(location):
    with open(location,"rb") as file:
        return pickle.load(file)
    
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
        
    