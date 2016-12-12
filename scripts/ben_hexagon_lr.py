#!/usr/bin/env python
from __future__ import print_function

import globs as G
import random
import numpy as np
import time
from collections import deque
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from convert_to_polar import reproject_image_into_polar
#%%
import OpenHexagonEmulator
from OpenHexagonEmulator import gameState
keys = np.array(['none', 'left_arrow', 'right_arrow'])



#%%
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3 # number of valid actions
GAMMA = 0.9 # decay rate of past observations
OBSERVATION = 2048. # timesteps to observe before training
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 2048.
BATCH = 2048
FRAME_PER_ACTION = 24
EPOCHS = 1000

#OpenHexagonEmulator.configure()

img_rows , img_cols = G.x_size, G.y_size

print('Resolution: ', img_rows, img_cols)
#Convert image into Black and white
img_channels = 2 # We stack 4 frames


def prepare_img(image):
    image = reproject_image_into_polar(image)[0]
    tmpImage = skimage.color.rgb2grey(image)
    thresh = skimage.filters.threshold_otsu(tmpImage)
    tmpImage = tmpImage > thresh
    tmpImage = skimage.transform.resize(tmpImage, (img_rows, img_cols))
    tmpImage = tmpImage.reshape(1, 1, tmpImage.shape[0], tmpImage.shape[1])
    return tmpImage

#==============================================================================
# CNN model based Q-learning - adapted for openhexagon
#==============================================================================
def trainNetwork():
    # open up a game state to communicate with emulator
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t = np.zeros((1, 1, img_rows,img_cols))
    s_t = np.zeros((1, img_channels, img_rows, img_cols))

    action_index = random.randrange(ACTIONS)

    t = 0
    alive = False
    while (len(D) < BATCH):
        if not alive:
            t = 0
            print('Resetting Game')
            OpenHexagonEmulator.press('enter')
            time.sleep(0.2)
            OpenHexagonEmulator.release('enter')
            time.sleep(3.5)


        current_run_frames = 0

        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            action_index = random.randrange(ACTIONS)

        # Run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = gameState(keys[action_index])

        alive = terminal == 0

        x_t1 = prepare_img(x_t1_colored)
        s_t1 = np.append(x_t1, s_t[:, :img_channels-1, :, :], axis=1)

        if alive:
            # store the transition in D
            D.append([s_t, action_index, r_t, s_t1, terminal])
        else:
            for i in range(60*3):
                if len(D) > 0:
                    D.pop()

        t += 1


    #sample a minibatch to train on
    minibatch = D

    inputs = [0]*BATCH
    targets = np.zeros(BATCH)

    #Now we do the experience replay
    for i in range(0, len(minibatch)):
        state_t = minibatch[i][0]
        action_t = minibatch[i][1]  # This is action index
        inputs[i:i + 1] = state_t
        targets[i] = action_t

    inputs = np.array(inputs)
    inputs.images.reshape((BATCH, -1))

    np.save('training', inputs)
    np.save('target', targets)

#==============================================================================
# Modified to run from Spyder Command window
#==============================================================================
def main():
    trainNetwork()
#==============================================================================

#==============================================================================
# Unmodified
#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================
