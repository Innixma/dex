# By Nick Erickson
# A3C Agent

import numpy as np
from memory import Memory
from metrics import Metrics
from brain_a3c import Brain
from data_utils import load_weights
import random

class Agent:
    def __init__(self, args, state_dim, action_dim, modelFunc=None, visualization=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.h = self.args.hyper
        self.epsilon = self.h.epsilon_init
        self.h.gamma_n = self.h.gamma ** self.h.memory_size
        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1
        self.mode = 'train'
        self.R = 0
        self.visualization = visualization
        
        self.metrics = Metrics()
        self.memory = Memory(self.h.memory_size, self.state_dim, 1)
        self.brain = Brain(self, modelFunc)
        
        load_weights(self)

    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
            
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            pr = self.brain.predict_p(np.array([s]))
            a = np.random.choice(self.action_dim, p=pr[0])
            return a
            #return np.argmax(self.brain.predict_p(np.array([s])))
    
    def act_v(self, s):
        pr, v = self.brain.predict(np.array([s]))
        if random.random() < self.epsilon:   
            return random.randrange(0, self.action_dim), v
        else:
            a = np.random.choice(self.action_dim, p=pr[0])
            return a, v
            #return np.argmax(p), v
            
    def observe(self, s, a, r, s_, t):
        self.memory.add_single(s, a, r, s_, t)
        self.update_epsilon()
        self.save_iterator += 1
        
    def replay(self, debug=True):
        self.replay_count += 1
        self.update_iterator += 1
        
        _, _, r, s_, t = self.memory.get_last()
        if t:
            s_ = None
        
        self.R = ( self.R + r * self.h.gamma_n ) / self.h.gamma

        if s_ is None:
            if self.memory.size < self.memory.max_size:
                self.memory.reset() # Don't train, R is inaccurate
            for i in range(self.memory.size, 0, -1):
                s, a, r, _, _ = self.memory.get_last_n(i)
                self.brain.train_augmented(s, a, self.R, None)
                self.R = ( self.R - r ) / self.h.gamma
            self.R = 0
            self.memory.reset()
            
        if self.memory.size >= self.memory.max_size:
            s, a, r, _, _ = self.memory.get_last_n(0)
            self.brain.train_augmented(s, a, self.R, s_)
            self.R = self.R - r  
