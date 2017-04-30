# By Nick Erickson
# Wrappers for Gym 

import numpy as np

import skimage.transform as transf
from skimage import color

import importlib

import gym

class Real_base_wrapper(object):
    def __init__(self, problem, module_name, class_name, screen_info):
        self.problem = problem
        self.module_name = module_name
        self.class_name = class_name
        self.module = importlib.import_module(module_name)
        self.class_func = getattr(self.module, self.class_name)
        self.env = self.class_func(screen_info)
    
    def start_game(self):
        self.env.start_game()    
        
    def step(self, a=0):
        s_, r, t = self.env.step(a)
        return s_, r, t
        
    def preprocess(self, s):
        return s

    def reset(self):
        s = self.env.start_game()
        return s
        
    def render(self):
        pass
        
    def state_dim(self):
        return self.env.state_dim      
        
    def action_dim(self):
        return self.env.action_dim
        pass
        

class Gym_base_wrapper(object):
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
    
        self.state_dim = list(self.reset().shape)
        self.action_dim = self.env.action_space.n
        
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

class Gym_rgb_wrapper(Gym_base_wrapper):
    def __init__(self, problem):
        super(Gym_rgb_wrapper, self).__init__(problem)
        self.state_dim = list(self.reset().shape[:-1])
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
        
class Gym_pong_wrapper(Gym_base_wrapper):
    def __init__(self, problem):
        super(Gym_rgb_wrapper, self).__init__(problem)
        self.state_dim = list(self.reset().shape[:-1])
        pass
    
    def step(self, a):
        s_, r, t, info = self.env.step(a)
        s_ = self.preprocess(s_)
        return s_, r, t, info
        
    def preprocess(self, s):
        s = color.rgb2gray(s).astype('float16')   
        s = s.reshape(list(s.shape) + [1])
        s = transf.downscale_local_mean(s, (2,2,1)) # Downsample
        s = s[17:-7,:,:]
        return s

    def reset(self):
        s = self.env.reset()
        s = self.preprocess(s)
        return s
