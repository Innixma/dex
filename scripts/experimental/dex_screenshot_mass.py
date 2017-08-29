#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with Tensorflow 1.1.0 and Keras v2.03

from __future__ import print_function

import cv2
import numpy as np

from utils.data_utils import loadMemory_direct


# Experimental
# import threading

class Screenshot_taker:
    def __init__(self, levels, scale=1):
        self.levels = levels
        self.scale = scale
        self.numLevels = len(self.levels)

    def format_results(self, images, width=10, scale=1):
        length = len(images)
        dim = list(images[0].shape[:-1])

        rows = int(length/width)+1

        if length/width == int(length/width):
            rows -= 1

        new_array = np.zeros([dim[0]*rows] + [dim[1]*width] + [3], dtype='uint8')
        #print(new_array.shape)

        for i in range(length):
            row = int(i/width)
            col = i % width

            new_array[row*dim[0]:(row+1)*dim[0], col*dim[1]:(col+1)*dim[1], :] = images[i]

        return new_array

    def take_screenshots(self):
        images = []
        for i in range(self.numLevels):

            level = self.levels[i]
            print(level)
            gather_dir = '../data/gather_' + level + '/'

            s, a, r, s_, t = loadMemory_direct(gather_dir)

            timelapse = np.arange(100, 350, 25)

            #print(np.mean(t))

            s_timelapse = s[timelapse]

            s_timelapse = s_timelapse[:,:,:,0]

            #print(s_timelapse.shape)
            s_timelapse *= 255
            s_timelapse = s_timelapse.astype('uint8')
            s_timelapse = self.rescale_images(s_timelapse, self.scale)
            s_timelapse = np.array(s_timelapse)
            s_timelapse = s_timelapse.reshape(list(s_timelapse.shape) + [1])
            #print(s_timelapse.shape)
            s_timelapse = list(s_timelapse)
            images.extend(s_timelapse)


        result = self.format_results(images)
        #result = self.rescale_images([result], self.scale)[0]

        print(result.shape)
        cv2.imwrite('outfile.png', result)

    def rescale_images(self, images, scale=1):
        numIter = len(images)

        dim = np.array(list(images[0].shape[:2]))
        #print(images[0].shape)
        new_dims = list(dim*scale)
        new_dims = tuple(new_dims)
        #print(new_dims)
        resized_images = []
        for i in range(numIter):
            resized = cv2.resize(images[i], new_dims, interpolation=cv2.INTER_AREA)
            resized_images.append(resized)
            #print(resized.shape)
        return resized_images

if __name__ == "__main__":

    levels = [
              'base_1',
              'base_2',
              'base_3',
              'rotation_1',
              'rotation_2',
              'rotation_3',
              'rotation_4',
              'rotation_5',
              'rotation_6',
              'rotation_7',
              'hexagon_1',
              'hexagon_2',
              'hexagon_3',
              'hexagon_4',
              'thinkfast'
              ]

    #levels = ['test', 'test']

    screenshot_taker = Screenshot_taker(levels, 2)

    screenshot_taker.take_screenshots()




