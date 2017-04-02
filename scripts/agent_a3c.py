# By Nick Erickson
# A3C Agent

import numpy as np
from memory import Memory
from metrics import Metrics
from brain_a3c import Brain
from data_utils import load_weights
import random

class Agent:
    def __init__(self, args, state_dim, action_dim, modelFunc=None):
        
        self.h = args.hyper
        self.h.gamma_n = self.h.gamma ** self.h.memory_size
        self.metrics = Metrics()
        self.memory = Memory(self.h.memory_size)
        self.brain = Brain(state_dim, action_dim, self.h, modelFunc)
        self.args = args
        self.epsilon = self.h.epsilon_init
        self.action_dim = action_dim
        self.state_dim = state_dim        
        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1
        self.mode = 'train'
        self.R = 0
        
        load_weights(self)

    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
            
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            return np.argmax(self.brain.predict_p(np.array([s])))
    
    def observe(self, sample):
        self.memory.add(sample)
        self.update_epsilon()
        self.save_iterator += 1
        
    def replay(self, debug=True): # Can make this even faster by giving arguments for last memory
        self.replay_count += 1
        self.update_iterator += 1
        
        _, _, r, s_, t = self.memory.D[-1]
        if t:
            s_ = None
        
        self.R = ( self.R + r * self.h.gamma_n ) / self.h.gamma

        if s_ is None:
            if self.memory.size < self.memory.max_size:
                self.memory.reset() # Don't train, R is inaccurate
            while self.memory.size > 0:
                s, a, r, _, _ = self.memory.popleft()
                self.brain.train_push(s, a, self.R, None)
                self.R = ( self.R - r ) / self.h.gamma
            self.R = 0
        if self.memory.size >= self.memory.max_size:
            s, a, r, _, _ = self.memory.popleft()
            self.brain.train_push(s, a, self.R, s_)
            self.R = self.R - r  
        
        if self.replay_count % 100 == 0:
            self.metrics.Q.append(0) # TODO: Save these better
            self.metrics.loss.append(0)
