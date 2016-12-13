#!/usr/bin/env python
from __future__ import print_function

import globs as G
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
#sys.path.append("game/")
#import wrapped_flappy_bird as game
import random
import numpy as np
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

from convert_to_polar import reproject_image_into_polar, plot_cart_image
#%%
import importlib
import OpenHexagonEmulator
import terminal_detection
#importlib.reload(OpenHexagonEmulator)
from OpenHexagonEmulator import gameState
keys = np.array(['none', 'left_arrow', 'right_arrow'])


#%%
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 10000. # timesteps to observe before training
EXPLORE = 1000000. # epochs over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 50000. # number of previous transitions to remember
BATCH = 32 #32 # size of minibatch
FRAME_PER_ACTION = 24
EPOCHS = 10000000

#OpenHexagonEmulator.configure()

img_rows , img_cols = G.x_size, G.y_size

print('Resolution: ', img_rows, img_cols)
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

#==============================================================================
# CNN model structure (Unmodified)
#==============================================================================
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model
#==============================================================================

def prepare_img(image):
    image = reproject_image_into_polar(image)[0]
    tmpImage = skimage.color.rgb2grey(image)
    tmpImage = tmpImage > 0
    tmpImage = skimage.transform.resize(tmpImage, (img_rows, img_cols))
    tmpImage = tmpImage.reshape(1, 1, tmpImage.shape[0], tmpImage.shape[1])
    tmpImage = tmpImage.astype(int)
    return tmpImage

#==============================================================================
# CNN model based Q-learning - adapted for openhexagon
#==============================================================================
def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    #game_state = game.GameState()

    with open('training_results.txt', 'w') as out_log:


        # store the previous observations in replay memory
        D = deque()
        episode_memory = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1

        x_t = np.zeros((1, 1, img_rows,img_cols))

        s_t = np.zeros((1, img_channels, img_rows, img_cols))

        if args['mode'] == 'Run':
            OBSERVE = 999999999    #We keep observe, never trai
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            model.load_weights("model.h5")
            adam = Adam(lr=1e-6)
            model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        run_count = -1
        saveIterator = 0
        saveThreshold = 10000
        survival_times = []
        num_epochs = 0
        epochs_flag = True
        action_index = 0


        alive = False
        while (epochs_flag):
            if not alive:
                num_actions = 0
                print('Resetting Game')
                OpenHexagonEmulator.press('enter')
                time.sleep(0.2)
                OpenHexagonEmulator.release('enter')
                time.sleep(3.2)

            run_count += 1
            run_start_t = t
            start_time = time.time()

            #run the selected action and observed next state and reward
            x_t1_colored, r_t, terminal = gameState(keys[action_index])

            alive = terminal == 0

            x_t1 = prepare_img(x_t1_colored)
            s_t1 = np.append(x_t1, s_t[:, :img_channels-1, :, :], axis=1)

            s_t = s_t1
            t = t + 1

            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    #print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    action_index = np.argmax(q)
                num_actions += 1

                # store the transition in D
                D.append([s_t, action_index, r_t, s_t1, terminal])
                if len(D) > REPLAY_MEMORY:
                    D.popleft()

            saveIterator += 1

            # Now Train!
            #only train if done observing
            if len(D) > OBSERVE:
                if not alive:
                    for i in range(num_actions):
                        loss = 0
                        Q_sa = 0
                        #sample a minibatch to train on
                        minibatch = random.sample(D, BATCH)
                        # minibatch[-1] = D[-1]

                        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
                        targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

                        #Now we do the experience replay
                        for i in range(0, len(minibatch)):
                            state_t = minibatch[i][0]
                            action_t = minibatch[i][1]   #This is action index
                            reward_t = minibatch[i][2]
                            state_t1 = minibatch[i][3]
                            terminal = minibatch[i][4]
                            # if terminated, only equals reward

                            inputs[i:i + 1] = state_t    #I saved down s_t

                            # Don't train on what we didn't observe
                            targets[i] = model.predict(state_t)
                            Q_sa = model.predict(state_t1)

                            if terminal:
                                targets[i, action_t] = reward_t
                            else:
                                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

                        # targets2 = normalize(targets)
                        loss += model.train_on_batch(inputs, targets)

                        if saveIterator >= saveThreshold:
                            saveIterator = 0
                            # save progress every 10000 iterations
                            print("Saving Model...")
                            model.save_weights("model.h5", overwrite=True)
                            with open("model.json", "w") as outfile:
                                json.dump(model.to_json(), outfile)

                        print("Q_MAX " , np.max(Q_sa), "/ L ", loss)
                        out_log.write('%.5f,%.5f\n' % (np.mean(Q_sa), loss))
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                        num_epochs += 1
                        epochs_flag = num_epochs <= EPOCHS
                        with open("log.txt", "a+") as outf:
                            outf.write('%d,%.10f,%.10f,%.10f\n' % (num_epochs, np.max(Q_sa), loss, time.time()-start_time))
                    print('Game terminated')
                    OpenHexagonEmulator.release(G.curKey)
                    time.sleep(0.1)
                    OpenHexagonEmulator.press('esc')
                    time.sleep(0.1)
                    OpenHexagonEmulator.release('esc')

    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
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

    args = {'mode' : 'Train'}
    # args = {'mode' : 'Run'}
    playGame(args)
#==============================================================================


#==============================================================================
# Unmodified
#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================
