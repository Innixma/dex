# By Nick Erickson
# Visualizing Data Testing

import numpy as np

import skimage.transform as transf
from skimage import color

import gym

class Gym_base_wrapper(object):
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        pass
    
    def step(self, a):
        s_, r, t, info = self.env.step(a)
        return s_, r, t, info
        
    def preprocess(self, s):
        return s

    def reset(self):
        s = self.env.reset()
        return s
        
    def render(self):
        self.env.render()
        
    def state_dim(self):
        return list(self.reset().shape)

class Gym_rgb_wrapper(Gym_base_wrapper):
    def __init__(self, problem):
        super(Gym_rgb_wrapper, self).__init__(problem)
        pass
    
    def step(self, a):
        s_, r, t, info = self.env.step(a)
        s_ = self.preprocess(s_)
        return s_, r, t, info
        
    def preprocess(self, s):
        s = color.rgb2gray(s).astype('float16')   
        s = s.reshape(list(s.shape) + [1])
        s = transf.downscale_local_mean(s, (2,2,1)) # Downsample
        return s

    def reset(self):
        s = self.env.reset()
        s = self.preprocess(s)
        return s
        
    def render(self):
        self.env.render()
