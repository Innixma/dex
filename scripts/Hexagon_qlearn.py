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

import os
RESULT_FOLDER = '/../results'
dir = os.path.dirname(__file__)
os.makedirs(dir + RESULT_FOLDER,exist_ok=True) # Generates results folder
RESULT_FOLDER_FULL = dir + RESULT_FOLDER
CUR_FOLDER = datetime.datetime.now().strftime('%G-%m-%d-%H-%M-%S')




keys = np.array(['none', 'left_arrow', 'right_arrow'])

t = 0

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

emulator = OpenHexagonEmulator.HexagonEmulator(G.application, [G.x_size, G.y_size], [G.x_zoom, G.y_zoom], [G.REWARD_ALIVE, G.REWARD_TERMINAL])

#ZZ = 0

"""
# Brain class concept from Jaromir Janisch, 2016
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
class Brain:
    def __init__(self, stateCnt, actionCnt, modelFunc=None):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel(modelFunc)
        self.model_ = self._createModel(modelFunc) 

    def _createModel(self, modelFunc=None):
        if modelFunc:
            model = modelFunc(self.stateCnt, self.actionCnt)
        else:
            model = Sequential()
    
            model.add(Dense(output_dim=64, activation='relu', input_dim=self.stateCnt))
            model.add(Dense(output_dim=self.actionCnt, activation='linear'))
    
            opt = RMSprop(lr=0.00025)
            model.compile(loss=hubert_loss, optimizer=opt)

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
"""

img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]

print('Resolution: ', img_rows, 'x,',img_cols, 'y')
#Convert image into Black and white

#==============================================================================
# CNN model structure (Base), Feb 27
#==============================================================================
def buildmodel_CNN():
    print("Now we build the model")
    
    model = Sequential()
    
    
    model.add(Convolution2D(16, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_cols,img_rows)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    
    # Why input_shape???? TODO
    #model.add(Flatten(input_shape=(img_channels,img_cols,img_rows)))
    model.add(Flatten())
    
    model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))

    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-4) # Base lr=1e-6
    model.compile(loss='mse',optimizer=adam)
    print("Finished building the model")
    print(model.summary())

    return model
#==============================================================================

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true
    return np.sqrt(1+np.square(err))-1

#==============================================================================
# CNN model structure (Base v2), Feb 27
#==============================================================================
def buildmodel_CNN_v2():
    print("Now we build the model")
    
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(img_channels,img_cols,img_rows)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    model.add(Activation('linear'))
    

    adam = Adam(lr=0.00025) # Base lr=1e-6
    model.compile(loss=hubert_loss,optimizer=adam) # Maybe try huber or mae??
    print("Finished building the model")
    print(model.summary())

    return model
#==============================================================================


# Converts image to grayscale, and forces image to proper dimensions
def prepareImage(image):
    
    tmpImage = skimage.color.rgb2gray(image)
    
    #img = smp.toimage(tmpImage)
    #smp.imsave('outfile1.png', img)
    
    # ----
    # Marked for deletion, image should already be in correct dimensions, else we have a problem.
    #tmpImage = skimage.transform.resize(tmpImage,(img_cols,img_rows)) 
    # ----
    
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
    

#==============================================================================
# CNN model based Q-learning - adapted for openhexagon
#==============================================================================
def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    #game_state = game.GameState()
    
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = emulator.gameState('enter')#game_state.frame_step(do_nothing)

    x_t = prepareImage(x_t)

    stacking = [x_t for i in range(img_channels)]
    s_t = np.stack(stacking, axis=0)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[3], s_t.shape[4])
    
    if args['directory'] == 'default':
        results_location = RESULT_FOLDER_FULL + '/' + CUR_FOLDER
    else:
        results_location = RESULT_FOLDER_FULL + '/' + args['directory']
    os.makedirs(results_location,exist_ok=True) # Generates results folder
    results_location = results_location + '/'
    
    # ------------------------------------------------------------
    # Loading Info
    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = 0
        print ("Now we load weight")
        model.load_weights(results_location + 'model.h5')
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    elif args['mode'] == 'Train_old': # Continue training old network
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        print ("Now we load weight")
        model.load_weights(results_location + 'model.h5')
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    else:                       # We go to training mode
        print('Training new network!')
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    # ------------------------------------------------------------
    
    mode = 'observe'
    global t
    t = 0
    t_saved = 0
    cur_saved = 0
    run_count = -1
    saveIterator = 0
    saveThreshold = INITIAL_SAVE_THRESHOLD
    timelapse = 1/CAPPED_FRAMERATE
    survival_times = []
    survival_times_last_10 = []
    survival_times_full_mean = []
    while (True):
        run_count += 1
        run_start_t = t
        alive = True
        emulator.press('enter')
        time.sleep(0.1)
        emulator.release('enter')
        start_time = time.time()
        current_run_frames = 0
        useRate = np.zeros([ACTIONS])
        cur_saved = 0
        while alive == True:
            if time.time() - start_time < (timelapse * current_run_frames): # Cap framerate
                time.sleep(timelapse - (time.time() % timelapse))
                
            a_t = np.zeros([ACTIONS])
            
            #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    #print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    action_index = np.argmax(q)
                    
                a_t[action_index] = 1
                
            #run the selected action and observed next state and reward        
            x_t1_colored, r_t, terminal = emulator.gameState(keys[action_index])
            
            if terminal: # Don't save terminal state itself, since it is pure white
                for i in range(NEG_REGRET_FRAMES):
                    if len(D) > i:
                        D[-1-i][2] = G.REWARD_TERMINAL/(i+1)
                if len(D) > 0:
                    D[-1][4] = 1 # Terminal State
            else:
                x_t1 = prepareImage(x_t1_colored)
                s_t1 = np.append(x_t1, s_t[:, :img_channels-1, :, :], axis=1)
            
                # Store the transition in D
                if current_run_frames > CAPPED_FRAMERATE*4: # Don't store early useless frames
                    t_saved += 1
                    cur_saved += 1
                    D.append([s_t, action_index, r_t, s_t1, terminal])
                        
                    # Reduce the epsilon gradually
                    if epsilon > FINAL_EPSILON and t_saved > OBSERVE:
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
                    if len(D) > REPLAY_MEMORY:
                        D.popleft()
                        
                    useRate = useRate + a_t 
        
                s_t = s_t1
                t = t + 1
                current_run_frames += 1
                saveIterator += 1

            if terminal == 1:
                # Lost!
                alive = 0    
            elif current_run_frames > 10000:
                # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                alive = 0
            
        # -----------------------------------
        # Reset keys and gamestate after loss
        end_time = time.time()
        #emulator.release(G.curKey) # Already done in emulator
        time.sleep(0.1)
        emulator.press('esc')
        time.sleep(0.1)
        emulator.release('esc')
        # -----------------------------------
        
        # -----------------------------------
        # Metrics Gathering
        if np.sum(useRate) != 0:
            useRate = useRate/np.sum(useRate)
        survival_time = end_time - start_time
        framerate = (t - run_start_t)/survival_time
        survival_times.append(survival_time)
        survival_times_last_10.append(np.mean(survival_times[-10:]))
        survival_times_full_mean.append(np.mean(survival_times))
        print('Run ' + str(run_count) + ' survived ' + "%.2f" % survival_time + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate])
        print('\tMean: %.2f' % np.mean(survival_times), 'Last 10: %.2f' % survival_times_last_10[-1], 'Max: %.2f' % np.max(survival_times), "Std: %.2f" % np.std(survival_times), "TS", t_saved, "E %.2f" % epsilon)
        # -----------------------------------
        
        # Now Train!
        # Only train if done observing
        if t_saved > OBSERVE:
            if mode == 'observe':
                mode = 'train'
                time.sleep(0.5)
            #sample a minibatch to train on
            for replay in range(1):
                loss = 0
                Q_sa = 0
                Q_sa_total = 0
                minibatch = random.sample(D, BATCH)
                
                # ---------------------------------------------------
                # Potential boost to performance, force learn terminal states
                for frame in range(NEG_REGRET_FRAMES):
                    minibatch.append(D[-frame-1])
                # ---------------------------------------------------
                
                inputs = np.zeros([len(minibatch), s_t.shape[1], s_t.shape[2], s_t.shape[3]])   #N, 80, 80, 4
                targets = np.zeros([inputs.shape[0], ACTIONS])                        #N, 3
                
                
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
    
                    targets[i] = model.predict(state_t)  # Hitting each button Q value
                    Q_sa = model.predict(state_t1)

                    Q_sa_total += np.max(Q_sa)

                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
    
                
                #print(targets)
                        
                # targets2 = normalize(targets)
                loss += model.train_on_batch(inputs, targets)
                Q_sa_total = Q_sa_total/len(minibatch)
                print("\tQ_MAX " , Q_sa_total, "/ L ", loss)
                
            if saveIterator >= saveThreshold:
                saveIterator = 0
                # save progress every 10000 iterations
                print("Saving Model...")
                model.save_weights(results_location + 'model.h5', overwrite=True)
                with open(results_location + 'model.json', "w") as outfile:
                    json.dump(model.to_json(), outfile)
    
            with open(results_location + 'log.txt', "a+") as outf:
                outf.write('%d,%.10f,%.10f,%.10f\n' % (run_count, np.max(Q_sa), loss, survival_time))
                
            graphHelper.graphSimple([np.arange(run_count+1),np.arange(run_count+1),np.arange(run_count+1)], [survival_times, survival_times_last_10, survival_times_full_mean], ['DQN', 'DQN_Last_10_Mean', 'DQN_Full_Mean'], 'DQN on Open Hexagon', 'Time(s)', 'Run', savefigName=results_location + 'DQN_graph')
        with open(results_location + 'DQN.txt', "a+") as outf:
            outf.write('%d,%.10f,%.10f,%.10f\n' % (run_count, survival_times[-1], survival_times_last_10[-1], survival_times_full_mean[-1]))    
            
        #print(np.arange(run_count))
        #print(survival_times)
        # Prep for next round
        time.sleep(0.2)
            
    print("Episode finished!")
    print("************************")
#==============================================================================

#==============================================================================
# Unmodified
#==============================================================================
def playGame(args):
    model = buildmodel_CNN_v2()
    trainNetwork(model,args)
#==============================================================================
    

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
    
    playGame(args)    
#==============================================================================


#==============================================================================
# Unmodified
#==============================================================================
if __name__ == "__main__":
    main() 
#==============================================================================
