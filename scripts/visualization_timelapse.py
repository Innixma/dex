

from data_utils import loadMemory_direct
from data_utils import load_weights
from agent_a3c import Agent
from parameters.hex import base_a3c
import models

import scipy.misc as smp
import cv2
import numpy as np

import visualization

from os import listdir
from os.path import isfile, join

results_folder = '../results/'
memory_folder = '../data/'


def get_models(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    only_models = [f for f in onlyfiles if 'model_' in f and '.h5' in f]
    only_models_names = [f[:-3] for f in only_models if 'frame' not in f]
    only_models_run = [int(f[6:]) for f in only_models_names]

    return only_models_names, only_models_run

def format_results(images, width=10, scale=1):
    length = len(images)
    dim = list(images[0].shape[:-1])

    rows = int(length/width)+1

    new_array = np.zeros([dim[0]*rows] + [dim[1]*width] + [3], dtype='uint8')
    print(new_array.shape)
    
    for i in range(length):
        row = int(i/width)
        col = i % width
        
        new_array[row*dim[0]:(row+1)*dim[0], col*dim[1]:(col+1)*dim[1], :] = images[i]

    return new_array
        




    
directory = 'hex_acer_hexreal_v1'
directory_mem = 'hex_acer_hexreal_v1'

args = base_a3c
agent_func = Agent

state_dim = [96, 96, 2]
action_dim = 3
brain = None


path = results_folder + directory

only_models_names, only_models_run = get_models(path)

agent = agent_func(args, state_dim, action_dim, getattr(models,args.model), visualization=True, brain=brain, idx=0)

agent.epsilon = 0
agent.h.epsilon_final = 0
agent.args.mode = 'run'
agent.args.directory = directory

path_mem = memory_folder + directory_mem + '/'

onlyfiles_mem = [f for f in listdir(path_mem) if isfile(join(path_mem, f))]
only_memory = [f for f in onlyfiles_mem if 'memory' in f and '.npz' in f]

extra = '_frame_2577'

s, a, r, s_, t = loadMemory_direct(path_mem, extra)

frame = np.arange(2000, 2500, 4)
frame = 2200
s1 = s[frame]
a1 = a[frame]
r1 = r[frame]
s_1 = s_[frame]
t1 = t[frame]

if len(s1.shape) != 4:
    s1 = s1.reshape([1] + list(s1.shape))
    #s1 = np.repeat(s1, 2, axis=3)
    
idx = [b[0] for b in sorted(enumerate(only_models_run),key=lambda i:i[1])]

#only_models_run = only_models_run[idx]
#only_models_names = only_models_names[idx]
only_models_run = [only_models_run[i] for i in idx]
only_models_names = [only_models_names[i] for i in idx]
text = [[str(i)] for i in only_models_run]

#frame = frame - np.min(frame)
#text = [str(i) for i in frame]




output_list = []
length = len(only_models_names)
length2 = int(length/2)

for i in range(length2, length):
#for i in range(1):
    name = only_models_names[i]
    curText = text[i]
    load_weights(agent, name)
    model = agent.brain.model
    _, _, overlayed_images, text_images = visualization.generate_saliceny_map(model, s1, show=False, text=curText)

    output_list.extend(text_images)
    
formatted = format_results(output_list)

print(only_models_run)

print(only_memory)

cv2.imshow('Saliency', formatted)
cv2.waitKey(0)
cv2.imwrite('outfile.png', formatted)
#img = smp.toimage(formatted, mode='P')
#smp.imsave('outfile.png', img)
