# By Nick Erickson
# Fetches image from gym environment

import cv2
import numpy as np

from environments.environment import EnvironmentGymRgb
from parameters import gym

args = gym.breakout_a3c

env = EnvironmentGymRgb(args.env, 0)

seed_img = env.env.reset()

print(seed_img.shape)

tmp_img = seed_img[:,:,0]
tmp_img = tmp_img.reshape(list(tmp_img.shape) + [1])
new_img = np.repeat(tmp_img, 3, 2)
new_img *= 255
new_img = np.clip(new_img, 0, 255)
new_img = new_img.astype('uint8')
#cv2.imshow('hey',new_img)
#cv2.waitKey(0)
cv2.imwrite('outfile.png', new_img)


