# By Nick Erickson
# Contains Save/Load Functions

import numpy as np
import pickle

# Saves memory, hyperparams, and screen info
def saveAll(agent, screen):
    saveMemory(agent)
    hyperfile = agent.data_location + 'hyper'
    screenfile = agent.data_location + 'screen'
    saveClass(agent.h, hyperfile)
    saveClass(screen, screenfile)
            
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
        
    