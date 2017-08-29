# By Nick Erickson
# Calls visualization.py

import numpy as np

import visualization
from agents import models
from agents.a3c.agent_a3c import Agent
from parameters.hex import base_a3c_load
from utils.data_utils import loadMemory_direct

args = base_a3c_load
state_dim = [96, 96, 2]
action_dim = 3
skip = 1
memory_location = '../data/' + args.directory + '/'
extra = '_frame_2577'

base_a3c_load.weight_override = 'model_frame_2577'
agent = Agent(base_a3c_load, state_dim, action_dim, modelFunc=models.model_mid_cnn, visualization=True)
s, a, r, s_, t = loadMemory_direct(memory_location, extra)

#prevVal = 0
imminent_idx = []
for i in range(0, t.shape[0], skip):
    #if prevVal == 1 and t[i] == 0:
    #    if i >= skip:
    imminent_idx.append(i)
    #prevVal = t[i]

imminent_s = s[imminent_idx]

#life_idx = []
#for i in range(10600, 15800+skip*5, skip):
#    life_idx.append(i-skip)

#life_s = s[life_idx]

model = agent.brain.model
print('Model loaded.')

heatmaps, heatmaps_c, overlayed_images, text_images = visualization.generate_saliceny_map(model, imminent_s, show=False)

real_images = []
for real in imminent_s:
    new_img = real[:,:,0]
    new_img = new_img.reshape(list(new_img.shape) + [1])
    new_img *= 255
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype('uint8')
    new_img = np.repeat(new_img, 3, 2)
    real_images.append(new_img)

print('rescaling')
real_images = visualization.rescale_images(real_images, scale=5)
heatmaps_c = visualization.rescale_images(heatmaps_c, scale=5)
overlayed_images = visualization.rescale_images(overlayed_images, scale=5)

print('concatenating')
composite_images = visualization.concat_3_videos(real_images, heatmaps_c, overlayed_images)

visualization.make_video(composite_images)
