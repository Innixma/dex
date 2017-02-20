#!/usr/bin/env python
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
from keras.optimizers import SGD , Adam
#%%
import importlib
import OpenHexagonEmulator
import terminal_detection
import graphHelper
#importlib.reload(OpenHexagonEmulator)
from OpenHexagonEmulator import gameState

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
CAPPED_FRAMERATE = 20 # Frames per second to process and act
OBSERVATION = 10000. # timesteps to observe before training
EXPLORE = 40000. # frames over which to anneal epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # 32 base # size of minibatch
FRAME_PER_ACTION = 1 # Number of frames inbetween actions, KEEP AT 1
INITIAL_SAVE_THRESHOLD = 10000 # Number of frames between saving the network
NEG_REGRET_FRAMES = 12 # Number of past frames to add regret to
#OpenHexagonEmulator.configure()

img_rows , img_cols = G.x_size_final, G.y_size_final

print('Resolution: ', img_rows, img_cols)
#Convert image into Black and white
img_channels = 1 #We stack img_channels frames (Default 4)




#==============================================================================
# CNN model structure (Unmodified)
#==============================================================================
def buildmodel():
    print("Now we build the model")
    
    model = Sequential()
    
    
    model.add(Convolution2D(16, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    
    
    
    
    #model.add(Convolution2D(64, 5, 5, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    
    #model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    #model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    
    #model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Convolution2D(16, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Convolution2D(16, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(64, 16, 16, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    #model.add(Activation('relu'))
    model.add(Flatten(input_shape=(img_channels,img_rows,img_cols)))
    #model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Dense(256, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    print(model.summary())
    G.model = model
    #print(model.layers)
    return model
#==============================================================================

def prepareImage(image):
    tmpImage = skimage.color.rgb2gray(image)
    #thresh = skimage.filters.threshold_otsu(tmpImage)
    #tmpImage = tmpImage > thresh
    tmpImage = skimage.transform.resize(tmpImage,(img_rows , img_cols))
            
    
    tmpImage = skimage.exposure.rescale_intensity(tmpImage, out_range=(0, 255))
    
    #print(tmpImage)
    #if t % 100 == 0:
    #    img = smp.toimage(tmpImage)
    #    #img.show()
    #    smp.imsave('outfile' + str(t) + '.png', img)
    #exit(1)
    #time.sleep(5)
    #print(tmpImage.shape)
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
    x_t, r_0, terminal = gameState('enter')#game_state.frame_step(do_nothing)
    #print(x_t.shape)
    x_t = prepareImage(x_t)
    #print(x_t.shape)
    
    #s_t = np.reshape(x_t, (1, 1, 1, img_rows, img_cols))
    stacking = [x_t for i in range(img_channels)]
    s_t = np.stack(stacking, axis=0)
    #print(s_t.shape)
    #s_t = np.stack((x_t), axis=0)
    
    #s_t = np.reshape(x_t, (1, 1, 1, img_rows, img_cols))
    #print(s_t.shape)
    
    #print(s_t.shape)
    #In Keras, need to reshape
    #print(s_t.shape)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[3], s_t.shape[4])
    #print(s_t.shape)
    #print(s_t.shape)
    
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
        epsilon = FINAL_EPSILON
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
        OpenHexagonEmulator.press('enter')
        time.sleep(0.1)
        OpenHexagonEmulator.release('enter')
        start_time = time.time()
        current_run_frames = 0
        useRate = np.zeros([ACTIONS])
        cur_saved = 0
        while alive == True:
            if time.time() - start_time < (timelapse * current_run_frames): # Cap framerate
                time.sleep(timelapse - (time.time() % timelapse))
            #print(s_t.shape)
            action_index = 0
            r_t = 0
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
            x_t1_colored, r_t, terminal = gameState(keys[action_index])
            
            x_t1 = prepareImage(x_t1_colored)
            
            s_t1 = np.append(x_t1, s_t[:, :img_channels-1, :, :], axis=1)
            #s_t1 = x_t1
            
            # store the transition in D
            if current_run_frames > CAPPED_FRAMERATE*4: # Don't store early useless frames
                t_saved += 1
                cur_saved += 1
                D.append([s_t, action_index, r_t, s_t1, terminal])
                if terminal == 1:
                    for i in range(NEG_REGRET_FRAMES):
                        D[-2-i][2] = G.REWARD_TERMINAL/(i+2)
                    #D[-(NEG_REGRET_FRAMES+1):-1][3] += G.REWARD_TERMINAL
                #We reduced the epsilon gradually
                if epsilon > FINAL_EPSILON and t_saved > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                if len(D) > REPLAY_MEMORY:
                    D.popleft()
            else:
                useRate = useRate + a_t
    
            s_t = s_t1
            t = t + 1
            current_run_frames += 1
            saveIterator += 1
    
            
    
            # print info
            state = ""
            if t_saved <= OBSERVE:
                state = "observe"
            elif t_saved > OBSERVE and t_saved <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            """
            if t % 1000 == 0:
                print("TS", t, "/ S", state, \
                    "/ E %.2f" % epsilon + " / A", action_index, "/ R", r_t)
            """
            if terminal == 1:
                # Lost!
                alive = 0
                
            if current_run_frames > 10000:
                # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                alive = 0
            
        # -----------------------------------
        # Reset keys and gamestate after loss
        end_time = time.time()
        OpenHexagonEmulator.release(G.curKey)
        time.sleep(0.1)
        OpenHexagonEmulator.press('esc')
        time.sleep(0.1)
        OpenHexagonEmulator.release('esc')
        terminal_detection.reset_globs()
        # -----------------------------------
        
        # -----------------------------------
        # Metrics Gathering
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
            
            #sample a minibatch to train on
            for replay in range(1):
                loss = 0
                Q_sa = 0
                minibatch = random.sample(D, BATCH)
                #for frame in range(NEG_REGRET_FRAMES):
                #    minibatch.append(D[-frame-1])
    
                inputs = np.zeros([len(minibatch), s_t.shape[1], s_t.shape[2], s_t.shape[3]])   #32, 80, 80, 4
                targets = np.zeros([inputs.shape[0], ACTIONS])                        #32, 2
                
                
                
                #Now we do the experience replay
                
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    # if terminated, only equals reward
                    
                    inputs[i:i + 1] = state_t    #I saved down s_t
    
                    targets[i] = model.predict(state_t)  # Hitting each buttom probability
                    Q_sa = model.predict(state_t1)
    
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
    
                
                #print(targets)
                        
                # targets2 = normalize(targets)
                loss += model.train_on_batch(inputs, targets)        
                print("\tQ_MAX " , np.max(Q_sa), "/ L ", loss)
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
    model = buildmodel()
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
