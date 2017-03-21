#!/usr/bin/env python

# Done: Target Network https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
# TODO: Prioritized Experience Replay
# TODO: Random agent saving memory/loading memory (Mostly done)
    
from __future__ import print_function

import globs as G
import argparse
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import scipy.misc as smp
import sys

import random
import numpy as np
import datetime
import time
from collections import deque

import pickle
import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam , RMSprop
#%%
import importlib
import OpenHexagonEmulator
import graphHelper

# Temp, for testing
import gym

import os
RESULT_FOLDER = '/../results'
dir = os.path.dirname(__file__)
os.makedirs(dir + RESULT_FOLDER,exist_ok=True) # Generates results folder
RESULT_FOLDER_FULL = dir + RESULT_FOLDER
CUR_FOLDER = datetime.datetime.now().strftime('%G-%m-%d-%H-%M-%S')

DATA_FOLDER = '/../data'
dir = os.path.dirname(__file__)
os.makedirs(dir + DATA_FOLDER,exist_ok=True) # Generates results folder
DATA_FOLDER_FULL = dir + DATA_FOLDER

#%%
    
class Metrics: # TODO: Save this to a pickle file?
    def __init__(self):
        self.survival_times = []
        self.survival_times_last_10 = []
        self.survival_times_full_mean = []
        self.Q = []
        self.loss = []
   
    def update(self, survival_time):
        self.survival_times.append(survival_time)
        self.survival_times_last_10.append(np.mean(self.survival_times[-10:]))
        self.survival_times_full_mean.append(np.mean(self.survival_times))

def hubert_loss(y_true, y_pred): # sqrt(1+a^2)-1
    err = y_pred - y_true
    return np.sqrt(1+np.square(err))-1

#==============================================================================
# CNN model structure (Base v3), Mar 18
#==============================================================================
def buildmodel_CNN_v3(state_dim, action_dim):
    
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=state_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_dim))
    model.add(Activation('linear'))
    

    adam = Adam(lr=0.00025)
    model.compile(loss=hubert_loss,optimizer=adam) # Maybe try huber or mae??

    return model
#==============================================================================  

# Class concept from Jaromir Janisch, 2016
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
class Brain:
    def __init__(self, state_dim, action_dim, modelFunc=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = self._createModel(modelFunc)
        self.model_ = self._createModel(modelFunc)
        
        self.updateTargetModel()
        
    def _createModel(self, modelFunc=None):
        if modelFunc:
            model = modelFunc(self.state_dim, self.action_dim)
        else:
            model = Sequential()
    
            model.add(Dense(output_dim=64, activation='relu', input_shape=self.state_dim))
            model.add(Dense(output_dim=self.action_dim, activation='linear'))
    
            opt = RMSprop(lr=0.00025)
            model.compile(loss=hubert_loss, optimizer=opt)
            #model.compile(loss='mse', optimizer=opt)
        print("Finished building the model")
        #print(model.summary())
        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        dim = [1] + self.state_dim
        return self.predict(s.reshape(dim), target=target).flatten()
        #return self.predict(s, target=target).flatten()
        
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

# TODO: Implement sum tree for prioritized learning
class Memory: # TODO: use maxlen argument in Deque?
    def __init__(self, max_size):
        self.D = deque()
        self.max_size = max_size
        self.size = 0
        self.total_saved = 0
        self.isFull = False
    
    def add(self, x):
        self.D.append(x)
        if self.size >= self.max_size:
            self.D.popleft()
            self.isFull = True
        else:
            self.size += 1
        self.total_saved += 1
        
    def removeLastN(self, n): # Remove last n instances
        if n > self.size:
            n = self.size
        for i in range(n):
            self.D.pop()
        self.size -= n
        self.total_saved -= n
        self.isFull = False
        
    def sample(self, batch_size):
        return random.sample(self.D, batch_size)

class RandomAgent:
    def __init__(self, hyperparams, args, state_dim, action_dim):
        self.h = hyperparams
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
        
        if self.args['directory'] == 'default':
            self.args['directory'] = CUR_FOLDER

        results_location = RESULT_FOLDER_FULL + '/' + self.args['directory']
        data_location = DATA_FOLDER_FULL + '/' + self.args['directory']
        os.makedirs(results_location,exist_ok=True) # Generates results folder
        os.makedirs(data_location,exist_ok=True) # Generates data folder
        self.results_location = results_location + '/'
        self.data_location = data_location + '/'
                
    def act(self, s):
        return random.randrange(0, self.action_dim)

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self):
        pass
        
class Agent:
    def __init__(self, hyperparams, args, state_dim, action_dim, modelFunc=None):
        self.h = hyperparams
        self.metrics = Metrics()
        self.memory = Memory(self.h.memory_size)
        self.brain = Brain(state_dim, action_dim, modelFunc)
        self.args = args
        self.epsilon = self.h.epsilon_init
        self.action_dim = action_dim
        self.state_dim = state_dim        
        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1
        self.mode = 'observe'
        
        self.load_weights()
        self.brain.updateTargetModel()
        
    def load_weights(self): # TODO: Update this function
        if self.args['directory'] == 'default':
            self.args['directory'] = CUR_FOLDER

        results_location = RESULT_FOLDER_FULL + '/' + self.args['directory']
        data_location = DATA_FOLDER_FULL + '/' + self.args['directory']
        os.makedirs(results_location,exist_ok=True) # Generates results folder
        os.makedirs(data_location,exist_ok=True) # Generates data folder
        self.results_location = results_location + '/'
        self.data_location = data_location + '/'
        
        if self.args['mode'] == 'Run':
            self.h.observe = 999999999    #We keep observe, never train
            self.epsilon = 0
            print ("Now we load weight")
            self.brain.model.load_weights(self.results_location + 'model.h5')
            #adam = Adam(lr=1e-6)
            #self.model.compile(loss='mse',optimizer=adam)
            print ("Weights loaded successfully")
        elif self.args['mode'] == 'Train_old': # Continue training old network
            self.h.observe = self.h.observe
            self.epsilon = self.h.epsilon_init
            print ("Now we load weight")
            self.brain.model.load_weights(self.results_location + 'model.h5')
            #adam = Adam(lr=1e-6)
            #self.model.compile(loss='mse',optimizer=adam)
            print ("Weights loaded successfully")
        else:                       # We go to training mode
            print('Training new network!')
            self.h.observe = self.h.observe
            self.epsilon = self.h.epsilon_init
     
    def save_weights(self):
        print("Saving Model...")
        self.brain.model.save_weights(self.results_location + 'model.h5', overwrite=True)
        with open(self.results_location + 'model.json', "w") as outfile:
            json.dump(self.brain.model.to_json(), outfile)
            
    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final and self.memory.total_saved > self.h.observe:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
        
    def update_agent(self):
        if self.update_iterator >= self.h.update_rate:
            self.update_iterator -= self.h.update_rate
            print('Updating Target Network')
            self.brain.updateTargetModel()
            
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            return np.argmax(self.brain.predictOne(s))
    
    def observe(self, sample):
        self.memory.add(sample)
        self.update_epsilon()
        self.save_iterator += 1
        #self.update_iterator += 1
        
    def replay(self, debug=True):
        self.replay_count += 1
        self.update_iterator += 1
        Q_sa_total = 0
        
        batch = self.memory.sample(self.h.batch)
        batchLen = len(batch)
        
        # ---------------------------------------------------
        # Potential boost to performance, force learn terminal states
        #for frame in range(self.h.neg_regret_frames):
        #    batch.append(self.memory.D[-frame-1])
        # ---------------------------------------------------
        
        states = np.array([x[0] for x in batch])
        states_ = np.array([x[3] for x in batch])
        
        targets = self.brain.predict(states)
        targets_ = self.brain.predict(states_, target=False) # Target Network!              
        pTarget_ = self.brain.predict(states_, target=True)                    
        Q_size = batchLen
        
        # TODO: Prioritized experience replay
        for i in range(0, batchLen):
            #state_t = batch[i][0]
            action_t = batch[i][1]
            reward_t = batch[i][2]
            #state_t1 = batch[i][3]
            terminal = batch[i][4]
            
            if terminal:
                Q_size -= 1
                targets[i, action_t] = reward_t
            else:
                Q_sa_total += np.max(targets_[i])
                #targets[i, action_t] = reward_t + self.h.gamma * np.max(targets_[i]) # Full DQN (Worse than double DQN)
                targets[i, action_t] = reward_t + self.h.gamma * pTarget_[i][np.argmax(targets_[i])]  # double DQN

        loss = self.brain.model.train_on_batch(states, targets) # Maybe do fit in future
        if Q_size == 0:
            Q_size = 1
        Q_sa_total = Q_sa_total/Q_size
        
        if debug:
            print("\tQ_MAX " , Q_sa_total, "/ L ", loss)
        
        if self.replay_count % 100 == 0:
            self.metrics.Q.append(Q_sa_total) # TODO: Save these better
            self.metrics.loss.append(loss)
            self.save_metrics_training() # TODO: move this
            
    def display_metrics(self, frame, useRate):
        if np.sum(useRate) != 0:
            useRate = useRate/np.sum(useRate)
        framerate = frame/self.metrics.survival_times[-1]
        print('Run ' + str(self.run_count) + ' survived ' + "%.2f" % self.metrics.survival_times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate])
        print('\tMean: %.2f' % np.mean(self.metrics.survival_times), 'Last 10: %.2f' % self.metrics.survival_times_last_10[-1], 'Max: %.2f' % np.max(self.metrics.survival_times), "Std: %.2f" % np.std(self.metrics.survival_times), "TS", self.memory.total_saved, "E %.2f" % self.epsilon)

    def save_metrics_training(self):
        graphHelper.graphSimple([np.arange(len(self.metrics.Q))], [self.metrics.Q], ['Q Value'], 'Q Value', 'Q Value', 'Run', savefigName=self.results_location + 'Q_graph')
        graphHelper.graphSimple([np.arange(len(self.metrics.loss))], [self.metrics.loss], ['Loss'], 'Loss', 'Loss', 'Run', savefigName=self.results_location + 'Loss_graph')        
        
    def save_metrics(self):
        with open(self.results_location + 'log.txt', "a+") as outf:
            outf.write('%d,%.10f,%.10f,%.10f\n' % (self.run_count, self.metrics.Q[-1], self.metrics.loss[-1], self.metrics.survival_times[-1]))
        # TODO: Remove these logs, just export the data directly
        graphHelper.graphSimple([np.arange(self.run_count+1),np.arange(self.run_count+1),np.arange(self.run_count+1)], [self.metrics.survival_times, self.metrics.survival_times_last_10, self.metrics.survival_times_full_mean], ['DQN', 'DQN_Last_10_Mean', 'DQN_Full_Mean'], 'DQN', 'Time(s)', 'Run', savefigName=self.results_location + 'DQN_graph')
        with open(self.results_location + 'DQN.txt', "a+") as outf:
            outf.write('%d,%.10f,%.10f,%.10f\n' % (self.run_count, self.metrics.survival_times[-1], self.metrics.survival_times_last_10[-1], self.metrics.survival_times_full_mean[-1]))    
            
        
class Environment_gym:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        
    def run(self, agent):
        s = self.env.reset()
        #print('hi')
        #print('there')
        #print('sir')
        #s = s.reshape([1] + list(s.shape))
        #s = s.reshape(1, 1, s.shape[0], s.shape[1])
        R = 0 
        
        while True:         
            #self.env.render()
            
            a = agent.act(s)
            #print('ok', a)
            s_, r, done, info = self.env.step(a)
            #s_ = s_.reshape([1] + list(s_.shape))
            #if done: # terminal state
                #print('done 2')
                #s_ = None
            #print('ok then')
            agent.observe( (s, a, r, s_, done) )
            if agent.mode == 'train':
                agent.replay(debug=False)
            #if agent.memory.size > agent.h.batch:
            #    agent.replay()            

            s = s_
            R += r
            #print('hi')
            if done:
                #print('done')
                return R, 1

class Environment_realtime:
    def __init__(self, emulator): # emulator is a working emulator class object
        self.env = emulator
        self.timelapse = 1
        #self.run_count = 0
        #self.saveThreshold = INITIAL_SAVE_THRESHOLD
        #self.timelapse = 1/CAPPED_FRAMERATE
        
    def framerate_check(self, start_time, frame):
        if time.time() - start_time < (self.timelapse * frame): # Cap framerate
            time.sleep(self.timelapse - (time.time() % self.timelapse))
        
    def init_run(self, img_channels):
        # get the first state by doing nothing and preprocess the image
        x_t, r_0, terminal = self.env.gameState()
    
        #x_t = prepareImage(x_t)
    
        stacking = [x_t for i in range(img_channels)]
        s_t = np.stack(stacking, axis=0)
    
        #In Keras, need to reshape
        #s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[3], s_t.shape[4])
        s_t = s_t.reshape(img_channels, s_t.shape[2], s_t.shape[3])
        return(s_t)
            
    def run(self, agent):
        frame = 0
        frame_saved = 0
        useRate = np.zeros([agent.action_dim])
        
        self.timelapse = 1/agent.h.framerate
        
        self.env.start_game()
        start_time = time.time()
        s_t = self.init_run(agent.h.img_channels)

        
        while self.env.alive:
            self.framerate_check(start_time, frame)
            action_index = agent.act(s_t)      
            x_t1, r_t, terminal = self.env.gameState(action_index)
            
            if terminal: # Don't save terminal state itself, since it is pure white
                for i in range(agent.h.neg_regret_frames):
                    if agent.memory.size > i:
                        agent.memory.D[-1-i][2] = self.env.reward_terminal/(i+1)
                if agent.memory.size > 0:
                    agent.memory.D[-1][4] = 1 # Terminal State
            else:
                s_t1 = np.append(x_t1, s_t[:agent.h.img_channels-1, :, :], axis=0)
                if frame > agent.h.framerate*4: # Don't store early useless frames
                    agent.observe([s_t, action_index, r_t, s_t1, terminal])
                    useRate[action_index] += 1
                    frame_saved += 1
        
                s_t = s_t1
                frame += 1
                
            if frame > 10000: # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                frame_saved = 0 # TODO: Still have to delete bad memories
                self.env.alive = False
                print('Deleting invalid memory...')
                agent.memory.removeLastN(10000)
        
        end_time = time.time()
        self.env.end_game()
        agent.run_count += 1
        
        agent.metrics.update(end_time-start_time)
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                time.sleep(1)
        
        return frame, useRate, frame_saved # Metrics

class Hyperparam:
    def __init__(
                 self,
                 framerate=40,
                 gamma=0.99,
                 batch=16,
                 observe=100,
                 explore=300,
                 epsilon_init=1.0,
                 epsilon_final=0.05,
                 memory_size=20000,
                 save_rate=10000,
                 neg_regret_frames=1,
                 img_channels=2,
                 update_rate=1000
                 ):
        
        self.framerate = framerate
        self.gamma = gamma
        self.batch = batch
        self.observe = observe
        self.explore = explore
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.memory_size = memory_size
        self.save_rate = save_rate
        self.neg_regret_frames = neg_regret_frames
        self.img_channels = img_channels
        self.update_rate = update_rate
        
class Screenparam:
    def __init__(self,
                 app=None,
                 size=[140,140],
                 zoom=[0,0]
                 ):
        self.app = app
        self.size = size
        self.zoom = zoom

def playGameGym(args, game, hyperparams=Hyperparam()):
    env = Environment_gym(game)
    state_dim  = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    hyperparams.img_channels = 1
    state_dim = [state_dim]
    agent = Agent(hyperparams, args, state_dim, action_dim)
    
    iteration = 0
    while (True):
        iteration += 1
        
        R, useRate = env.run(agent)
        
        agent.update_agent()
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                print('training')
                time.sleep(0.5)
                
        if agent.mode == 'train':
            if iteration % 10 == 0:
                agent.replay(debug=True)
            else:
                agent.replay(debug=False)
            
            if iteration % 10 == 0:
                S = np.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
                S = S.reshape([1] + list(S.shape))
                pred = agent.brain.predictOne(S)
                print(pred[0])
                sys.stdout.flush()
                print("Step:", agent.memory.total_saved)
                print("Total reward:", R)
            
        #agent.display_metrics(frame, useRate)
        
        #if agent.mode == 'train': # Fix this later, not correct
        #    agent.save_metrics()
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            agent.save_weights()
  
def playGameReal(args, screen=Screenparam(), hyperparams=Hyperparam()):
    emulator = OpenHexagonEmulator.HexagonEmulator(
                                                   screen.app,
                                                   screen.size,
                                                   screen.zoom
                                                  )
    img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
    img_channels = hyperparams.img_channels
    state_dim = [img_channels, img_rows, img_cols]
    action_dim = emulator.action_dim
    
    agent = Agent(hyperparams, args, state_dim, action_dim, buildmodel_CNN_v3)
    
    if args['data'] != 'default':
        # Load Memory
        loadMemory(agent, args['data'])
        
        agent.mode = 'train'
        loaded_replays = int(agent.memory.size/2)
        print('Running', loaded_replays, 'replays')
        # Train on loaded memory
        for i in range(loaded_replays):
            agent.update_agent()
            if i % 5000 == 0:
                print(i, '/', loaded_replays)
            if i % 100 == 0:
                agent.replay(debug=True)
            else:
                agent.replay(debug=False)
    
    time.sleep(1)
    
    env = Environment_realtime(emulator)
    while (True):
        frame, useRate, frame_saved = env.run(agent)
        
        
        
        if agent.mode == 'train':
            print('Running', frame_saved, 'replays')
            for i in range(frame_saved):
                agent.update_agent()
                if i % 100 == 0:
                    agent.replay(debug=True)
                else:
                    agent.replay(debug=False)
            
        agent.display_metrics(frame, useRate)
        
        if agent.mode == 'train': # Fix this later, not correct
            agent.save_metrics()
            agent.save_metrics_training()
            
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            agent.save_weights()

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
        
            
def gatherMemory(args, screen=Screenparam(), hyperparams=Hyperparam()):
    emulator = OpenHexagonEmulator.HexagonEmulator(
                                                   screen.app,
                                                   screen.size,
                                                   screen.zoom
                                                  )
    img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
    img_channels = hyperparams.img_channels
    state_dim = [img_channels, img_rows, img_cols]
    action_dim = emulator.action_dim
    
    agent = RandomAgent(hyperparams, args, state_dim, action_dim)
    
    env = Environment_realtime(emulator)
    print('Gathering', agent.memory.max_size, 'states:')
    while (True):
        frame, useRate, frame_saved = env.run(agent)
        
        #agent.display_metrics(frame, useRate)
        print(agent.memory.size, '/', agent.memory.max_size)
        
        
        if agent.memory.isFull:
            return saveAll(agent, screen)
            
            

    
        
hexagonHyper1 = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=2000,
                             explore=2000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=50000,
                             save_rate=10000,
                             neg_regret_frames=1,
                             img_channels=2,
                             update_rate=1000
                           )
        
hexagonScreen1 = Screenparam(
                             app='Open Hexagon 1.92 - by vittorio romeo',
                             size=[140,140],
                             zoom=[28,18]
                             )

cartHyper1 = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=100000,
                             explore=30000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=100000,
                             save_rate=100000,
                             neg_regret_frames=0,
                             img_channels=1,
                             update_rate=1000
                           )

gatherHyper1 = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=0,
                             explore=10000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=50000,
                             save_rate=10000,
                             neg_regret_frames=1,
                             img_channels=2,
                             update_rate=1000
                           )

def runSimulation(args):
     
    if args['type'] == 'real':
        hyper = hexagonHyper1
        screen = hexagonScreen1
        
        if args['hyper'] != 'default':
            hyper = loadClass(args['hyper'])
            
        if args['screen'] != 'default':
            screen = loadClass(args['screen'])
            
        playGameReal(args, screen, hyper)
        
    elif args['type'] == 'gym':
        hyper = cartHyper1
        game = 'CartPole-v0'
        
        if args['hyper'] != 'default':
            hyper = loadClass(args['hyper'])
        if args['game'] != 'default':
            game = args['game']

        playGameGym(args, game, hyper)
        
    elif args['type'] == 'memory':
        hyper = gatherHyper1
        screen = hexagonScreen1
        
        if args['hyper'] != 'default':
            hyper = loadClass(args['hyper'])
            
        if args['screen'] != 'default':
            screen = loadClass(args['screen'])
            
        gatherMemory(args, screen, hyper)
    
    else:
        pass
    
    
    #gatherMemory(args, hexagonScreen1, gatherHyper1)
    #loadMemory()

#==============================================================================
# Modified to run from Spyder Command window
#==============================================================================
def main():    
    #parser = argparse.ArgumentParser(description='Description of your program')
    #parser.add_argument('-m','--mode', help='Train / Run', required=True)    
    #args = vars(parser.parse_args())
    
    hex_base_data = 'hex_base_40fps_50k'
    
    args = {}
    
    args['mode'] = 'Train'
    args['game'] = 'default'
    args['type'] = 'real' # gym, real, memory
    #args['data'] = 'default'
    args['data'] = DATA_FOLDER_FULL + '/' + hex_base_data + '/'
    args['screen'] = 'default'
    args['hyper'] = 'default'
    args['directory'] = 'default'
    
    runSimulation(args)

    #args['mode'] = 'Run'
    #args['directory'] = 'name_of_file'
    
    # TODO: Use finally clause to save stuff
#==============================================================================


#==============================================================================
# Unmodified
#==============================================================================
if __name__ == "__main__":
    main() 
#==============================================================================
