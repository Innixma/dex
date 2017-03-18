#!/usr/bin/env python

# TODO: Target Network https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/

from __future__ import print_function

import globs as G
import argparse
import skimage as skimage
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

keys = np.array(['none', 'left_arrow', 'right_arrow'])

#%%
GAME = 'open_hexagon' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
CAPPED_FRAMERATE = 18 # Frames per second to process and act
OBSERVATION = CAPPED_FRAMERATE*600 # timesteps to observe before training
EXPLORE = CAPPED_FRAMERATE*2000 # frames over which to anneal epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
REPLAY_MEMORY = CAPPED_FRAMERATE*1000 # number of previous transitions to remember
BATCH = CAPPED_FRAMERATE*5 # 32 base # size of minibatch
FRAME_PER_ACTION = 1 # Number of frames inbetween actions, KEEP AT 1
INITIAL_SAVE_THRESHOLD = 10000 # Number of frames between saving the network
NEG_REGRET_FRAMES = 1 #int(CAPPED_FRAMERATE/6) # Number of past frames to add regret to
img_channels = 2 #We stack img_channels frames (Default 4)


#ZZ = 0


#emulator = OpenHexagonEmulator.HexagonEmulator(G.application, [G.x_size, G.y_size], [G.x_zoom, G.y_zoom], [G.REWARD_ALIVE, G.REWARD_TERMINAL])
#img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
#print('Resolution: ', img_rows, 'x,',img_cols, 'y')


#hyperparams = Hyperparam()
    
class Metrics:
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

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
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
    

    adam = Adam(lr=0.00025) # Base lr=1e-6
    model.compile(loss=hubert_loss,optimizer=adam) # Maybe try huber or mae??

    return model
#==============================================================================


# Converts image to grayscale, and forces image to proper dimensions
def prepareImage(image):
    
    tmpImage = skimage.color.rgb2gray(image)
    
    #global ZZ
    #ZZ += 1
    #if ZZ > 300:
    #    img = smp.toimage(tmpImage)
    #    smp.imsave('outfile' + str(ZZ % 36) + '.png', img)    
    
    # Following line commented out Feb 25 2017, due to potential issues caused.
    #tmpImage = skimage.exposure.rescale_intensity(tmpImage, out_range=(0, 255))
    
    # ----
    # NEW!!! Feb 28: Normalize pixels
    tmpImage = tmpImage.astype('float32') / 128 - 1
    # ----
    
    tmpImage = tmpImage.reshape(1, 1, tmpImage.shape[0], tmpImage.shape[1])
    
    return tmpImage
    

# Brain class concept from Jaromir Janisch, 2016
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
    
            model.add(Dense(output_dim=64, activation='relu', input_dim=self.state_dim))
            model.add(Dense(output_dim=self.action_dim, activation='linear'))
    
            opt = RMSprop(lr=0.00025)
            model.compile(loss=hubert_loss, optimizer=opt)

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
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    
class Memory:
    def __init__(self, max_size):
        self.D = deque()
        self.max_size = max_size
        self.size = 0
        self.total_saved = 0
    
    def add(self, x):
        self.D.append(x)
        if self.size >= self.max_size:
            self.D.popleft()
        else:
            self.size += 1
        self.total_saved += 1
        
    def sample(self, batch_size):
        return random.sample(self.D, batch_size)
    
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
        self.save_iterator = 0
        self.mode = 'observe'
        
        self.load_weights()
        
    def load_weights(self):
        if self.args['directory'] == 'default':
            results_location = RESULT_FOLDER_FULL + '/' + CUR_FOLDER
        else:
            results_location = RESULT_FOLDER_FULL + '/' + self.args['directory']
        os.makedirs(results_location,exist_ok=True) # Generates results folder
        results_location = results_location + '/'
        if self.args['mode'] == 'Run':
            self.h.observe = 999999999    #We keep observe, never train
            self.epsilon = 0
            print ("Now we load weight")
            self.brain.model.load_weights(results_location + 'model.h5')
            #adam = Adam(lr=1e-6)
            #self.model.compile(loss='mse',optimizer=adam)
            print ("Weights loaded successfully")
        elif self.args['mode'] == 'Train_old': # Continue training old network
            self.h.observe = self.h.observe
            self.epsilon = self.h.epsilon_init
            print ("Now we load weight")
            self.brain.model.load_weights(results_location + 'model.h5')
            #adam = Adam(lr=1e-6)
            #self.model.compile(loss='mse',optimizer=adam)
            print ("Weights loaded successfully")
        else:                       # We go to training mode
            print('Training new network!')
            self.h.observe = self.h.observe
            self.epsilon = INITIAL_EPSILON
        self.results_location = results_location
     
    def save_weights(self):
        print("Saving Model...")
        self.brain.model.save_weights(self.results_location + 'model.h5', overwrite=True)
        with open(self.results_location + 'model.json', "w") as outfile:
            json.dump(self.brain.model.to_json(), outfile)
            
    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final and self.memory.total_saved > self.h.observe:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim)
        else:
            return np.argmax(self.brain.predictOne(s))
    
    def observe(self, sample):
        pass
    
    def replay(self):
        #sample a minibatch to train on
        loss = 0
        Q_sa = 0
        Q_sa_total = 0
        minibatch = self.memory.sample(self.h.batch)
        
        # ---------------------------------------------------
        # Potential boost to performance, force learn terminal states
        for frame in range(self.h.neg_regret_frames):
            minibatch.append(self.memory.D[-frame-1])
        # ---------------------------------------------------
        
        inputs = np.zeros([len(minibatch), self.state_dim[0], self.state_dim[1],self.state_dim[2]])   #N, 80, 80, 4
        targets = np.zeros([inputs.shape[0], self.action_dim])                        #N, 3
        
        
        # TODO: Prioritized experience replay
        #Now we do the experience replay
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]   #This is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]
            # if terminated, only equals reward
            
            inputs[i:i + 1] = state_t

            targets[i] = self.brain.model.predict(state_t)  # Hitting each button Q value
            Q_sa = self.brain.model.predict(state_t1)

            Q_sa_total += np.max(Q_sa)

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.h.gamma * np.max(Q_sa)

        
        #print(targets)
                
        # targets2 = normalize(targets)
        loss += self.brain.model.train_on_batch(inputs, targets)
        Q_sa_total = Q_sa_total/len(minibatch)
        print("\tQ_MAX " , Q_sa_total, "/ L ", loss)
        
        self.metrics.Q.append(Q_sa_total)
        self.metrics.loss.append(loss)
        
    def display_metrics(self, frame, useRate):
        if np.sum(useRate) != 0:
            useRate = useRate/np.sum(useRate)
        framerate = frame/self.metrics.survival_times[-1]
        print('Run ' + str(self.run_count) + ' survived ' + "%.2f" % self.metrics.survival_times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate])
        print('\tMean: %.2f' % np.mean(self.metrics.survival_times), 'Last 10: %.2f' % self.metrics.survival_times_last_10[-1], 'Max: %.2f' % np.max(self.metrics.survival_times), "Std: %.2f" % np.std(self.metrics.survival_times), "TS", self.memory.total_saved, "E %.2f" % self.epsilon)

    def save_metrics(self):
        with open(self.results_location + 'log.txt', "a+") as outf:
            outf.write('%d,%.10f,%.10f,%.10f\n' % (self.run_count, self.metrics.Q[-1], self.metrics.loss[-1], self.metrics.survival_times[-1]))
                
        graphHelper.graphSimple([np.arange(self.run_count+1),np.arange(self.run_count+1),np.arange(self.run_count+1)], [self.metrics.survival_times, self.metrics.survival_times_last_10, self.metrics.survival_times_full_mean], ['DQN', 'DQN_Last_10_Mean', 'DQN_Full_Mean'], 'DQN on Open Hexagon', 'Time(s)', 'Run', savefigName=self.results_location + 'DQN_graph')
        with open(self.results_location + 'DQN.txt', "a+") as outf:
            outf.write('%d,%.10f,%.10f,%.10f\n' % (self.run_count, self.metrics.survival_times[-1], self.metrics.survival_times_last_10[-1], self.metrics.survival_times_full_mean[-1]))    
            
        
class Environment_gym:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        
    def run(self, agent):
        s = self.env.reset()
        R = 0 

        while True:            
            # self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_, done) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

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
    
        x_t = prepareImage(x_t)
    
        stacking = [x_t for i in range(img_channels)]
        s_t = np.stack(stacking, axis=0)
    
        #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[3], s_t.shape[4])

        return(s_t)
            
    def run(self, agent):
        frame = 0
        useRate = np.zeros([agent.brain.action_dim])
        
        self.timelapse = 1/agent.h.framerate
        
        self.env.start_game()
        start_time = time.time()
        s_t = self.init_run(agent.h.img_channels)

        while self.env.alive:
            self.framerate_check(start_time, frame)
            
            if random.random() <= agent.epsilon: # Choose an action epsilon greedy
                action_index = random.randrange(agent.brain.action_dim)
            else:
                q = agent.brain.model.predict(s_t)
                action_index = np.argmax(q)
                
            #run the selected action and observed next state and reward        
            x_t1_colored, r_t, terminal = self.env.gameState(keys[action_index])
            
            if terminal: # Don't save terminal state itself, since it is pure white
                for i in range(agent.h.neg_regret_frames):
                    if agent.memory.size > i:
                        agent.memory.D[-1-i][2] = self.env.reward_terminal/(i+1)
                if agent.memory.size > 0:
                    agent.memory.D[-1][4] = 1 # Terminal State
            else:
                x_t1 = prepareImage(x_t1_colored)
                s_t1 = np.append(x_t1, s_t[:, :agent.h.img_channels-1, :, :], axis=1)
                if frame > agent.h.framerate*4: # Don't store early useless frames
                    agent.memory.add([s_t, action_index, r_t, s_t1, terminal])
                    agent.update_epsilon() # Reduce the epsilon gradually
                    agent.save_iterator += 1
                    useRate[action_index] += 1 
        
                s_t = s_t1
                frame += 1
            if frame > 10000: # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                self.env.alive = False
        
        end_time = time.time()
        self.env.end_game()
        agent.run_count += 1
        
        agent.metrics.update(end_time-start_time)
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                time.sleep(0.5)
        
        return frame, useRate # Metrics

class Hyperparam:
    def __init__(
                 self,
                 framerate=30,
                 gamma=0.99,
                 batch=16,
                 observe=100,
                 explore=300,
                 epsilon_init=1.0,
                 epsilon_final=0.05,
                 memory_size=20000,
                 save_rate=10000,
                 neg_regret_frames=1,
                 img_channels=2
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
        
        
class Screenparam:
    def __init__(self,
                 app=None,
                 size=[140,140],
                 zoom=[0,0]
                 ):
        self.app = app
        self.size = size
        self.zoom = zoom


#==============================================================================
# Unmodified
#==============================================================================
#def playGame(args):
#    model = buildmodel_CNN_v2()
#    trainNetwork(model,args)
#==============================================================================
    
def playGame2(args, screen=Screenparam(), hyperparams=Hyperparam()):
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
    
    env = Environment_realtime(emulator)
    while (True):
        frame, useRate = env.run(agent)
        
        if agent.mode == 'train':
            agent.replay()
            
        agent.display_metrics(frame, useRate)
        
        if agent.mode == 'train': # Fix this later, not correct
            agent.save_metrics()
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            agent.save_weights()

hexagonHyper1 = Hyperparam(
                             framerate=30,
                             gamma=0.99,
                             batch=16,
                             observe=1000,
                             explore=3000,
                             epsilon_init=1.0,
                             epsilon_final=0.05,
                             memory_size=20000,
                             save_rate=10000,
                             neg_regret_frames=1,
                             img_channels=2
                           )
        
hexagonScreen1 = Screenparam(
                             app='Open Hexagon 1.92 - by vittorio romeo',
                             size=[140,140],
                             zoom=[28,18]
                             )
            
#==============================================================================
# Modified to run from Spyder Command window
#==============================================================================
def main():    
    #parser = argparse.ArgumentParser(description='Description of your program')
    #parser.add_argument('-m','--mode', help='Train / Run', required=True)    
    #args = vars(parser.parse_args())
    
    args = {}
    args['mode'] = 'Train'
    args['directory'] = 'default'
    
    #args['mode'] = 'Run'
    #args['directory'] = 'name_of_file'
    
    #playGame(args)
    playGame2(args, hexagonScreen1, hexagonHyper1)    
#==============================================================================


#==============================================================================
# Unmodified
#==============================================================================
if __name__ == "__main__":
    main() 
#==============================================================================
