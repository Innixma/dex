

from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

import visualization
from agents import models
from agents.a3c.agent_a3c import Agent
from parameters.hex import base_a3c
from utils.data_utils import loadMemory_direct
from utils.data_utils import load_weights

results_folder = '../results/'
memory_folder = '../data/'


def get_models(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    only_models = [f for f in onlyfiles if 'model_' in f and '.h5' in f]
    only_models_names = [f[:-3] for f in only_models if 'frame' not in f and 'max' not in f]

    print(only_models_names)
    only_models_run = [int(f[6:]) for f in only_models_names]

    return only_models_names, only_models_run

def format_results(images, width=10, scale=1):
    length = len(images)
    dim = list(images[0].shape[:-1])

    rows = int(length/width)+1

    if length/width == int(length/width):
            rows -= 1

    new_array = np.zeros([dim[0]*rows] + [dim[1]*width] + [3], dtype='uint8')
    print(new_array.shape)

    for i in range(length):
        row = int(i/width)
        col = i % width

        new_array[row*dim[0]:(row+1)*dim[0], col*dim[1]:(col+1)*dim[1], :] = images[i]



    return new_array

def gen_saliency(directory, directory_mem, txt):

    args = base_a3c
    agent_func = Agent

    state_dim = [42, 42, 2]
    action_dim = 3
    brain = None


    path = results_folder + directory

    #only_models_names, only_models_run = get_models(path)
    only_models_names = ['model_max']
    only_models_run = [txt]

    agent = agent_func(args, state_dim, action_dim, getattr(models, args.model), visualization=True, brain=brain, idx=0)

    agent.epsilon = 0
    agent.h.epsilon_final = 0
    agent.args.mode = 'run'
    agent.args.directory = directory
    agent.args.weight_override = 'model_max'

    path_mem = memory_folder + directory_mem + '/'
    extra = ''

    s, a, r, s_, t = loadMemory_direct(path_mem, extra)

    frame = np.arange(2400, 2500, 4)
    frame = 2400
    s1 = s[frame]
    a1 = a[frame]
    r1 = r[frame]
    s_1 = s_[frame]
    t1 = t[frame]

    if len(s1.shape) != 4:
        s1 = s1.reshape([1] + list(s1.shape))
        #s1 = np.repeat(s1, 2, axis=3)

    #idx = [b[0] for b in sorted(enumerate(only_models_run),key=lambda i:i[1])]
    #only_models_run = [only_models_run[i] for i in idx]
    #only_models_names = [only_models_names[i] for i in idx]


    #text = [[str(i)] for i in only_models_run]
    text = [[str(i)] for i in only_models_run]
    #frame = frame - np.min(frame)
    #text = [str(i) for i in frame]




    output_list = []
    length = len(only_models_names)
    #length2 = int(length/2)

    for i in range(0, length):
    #for i in range(1):
        name = only_models_names[i]
        curText = text[i]
        load_weights(agent, name)
        model = agent.brain.model
        _, _, overlayed_images, text_images = visualization.generate_saliceny_map(model, s1, show=False, text=curText)

        output_list.extend(text_images)

    #scale = 8
    #rescale_images = visualization.rescale_images(output_list, scale)
    #formatted = format_results(output_list)



    print(only_models_run)
    return output_list

    #cv2.imshow('Saliency', formatted)
    #cv2.waitKey(0)
    #cv2.imwrite('outfile.png', formatted)
    #img = smp.toimage(formatted, mode='P')
    #smp.imsave('outfile.png', img)
"""
levels = [
              'rotation_1',
              'rotation_2',
              'rotation_3',
              'rotation_4',
              'rotation_5',
              'rotation_6',
              'rotation_7'
             ]
             
levelpairs = [
              ['rotation_1','rotation_2'],
              ['rotation_2','rotation_3'],
              ['rotation_3','rotation_4'],
              ['rotation_4','rotation_5'],
              ['rotation_5','rotation_6'],
              ['rotation_6','rotation_7']
             ]
"""

levels = [
              'base_1',
              'base_2',
              'base_3'
             ]

levelpairs = [
              ['base_1','base_2'],
              ['base_2','base_3']
             ]

directories = ['experiment_v1/trained_' + l for l in levels]
directories2 = ['experiment_v1/trained_' + l[0] + '_' + l[1] for l in levelpairs]

directories.extend(directories2)
directory_mem = ['experiment_v1/test_base_3' for d in directories]

output_list = []

nameList = [1,2,3,4,5,6,7,2,3,4,5,6,7]
nameList = [1,2,3,2,3]
for i in range(len(directories)):
    output_list.extend(gen_saliency(directories[i], directory_mem[i], str(nameList[i])))

scale = 8
rescale_images = visualization.rescale_images(output_list, scale)
formatted = format_results(rescale_images, 3)
cv2.imshow('Saliency', formatted)
cv2.waitKey(0)
cv2.imwrite('outfile.png', formatted)
