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
        self.metrics = Metrics()
        self.memory = Memory(self.h.memory_size)
        self.brain = Brain(state_dim, action_dim, modelFunc)
        self.args = args
        self.epsilon = self.h.epsilon_init
        self.action_dim = action_dim
        self.state_dim = state_dim        
        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1
        self.mode = 'observe'
        self.R = 0
        
        load_weights(self)

    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final and self.memory.total_saved > self.h.observe:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
            
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            s = np.array([s])
            return np.argmax(self.brain.predict_p(s))
    
    def observe(self, sample):
        #print(self.memory.size)
        self.memory.add(sample)
        #print(self.memory.size)
        self.update_epsilon()
        self.save_iterator += 1
        #self.update_iterator += 1
        
    def replay(self, debug=True):
        self.replay_count += 1
        self.update_iterator += 1
        
        #batch = self.memory.sample(self.h.batch)
        #batch = self.memory.sample(1)
        #print(self.memory.size)
        batch = self.memory.D[-1]
        if batch[4]:
            z = None
        else:
            z = batch[3]
        self.train(batch[0], batch[1], batch[2], z)
        #self.brain.train_push(batch[0], batch[1], batch[2], z)
        #batchLen = len(batch)
        #for x in batch:
            #self.brain.train_push(x[0], x[1], x[2], x[3])
        #self.brain.optimize()
        
        if self.replay_count % 100 == 0:
            self.metrics.Q.append(0) # TODO: Save these better
            self.metrics.loss.append(0)
            
    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _, _  = memory[0]
            _, _, _, s_, _ = memory[n-1]

            return s, a, self.R, s_

        #a_cats = np.zeros(self.action_dim)    # turn action into one-hot representation
        #a_cats[a] = 1 

        #self.memory.append( (s, a_cats, r, s_) )

        self.R = ( self.R + r * self.h.gamma_n ) / self.h.gamma # Nick: Is this wrong?
        #print(self.R)
        if s_ is None:
            #n = self.memory.size
            while self.memory.size > 0:
                #n = len(self.memory)
                n = self.memory.size
                s, a, r, s_ = get_sample(self.memory.D, n) # Possibly really slow # TODO: fix
                
                self.brain.train_push(s, a, r, None) # TODO: Fixed this by adding None
                #print(self.R)
                self.R = ( self.R - self.memory.D[0][2] ) / self.h.gamma
                self.memory.removeFirstN(1)
                #print(self.memory.size)
                #print(self.R)
            self.R = 0
            #print('hi')
        if self.memory.size >= self.h.n_step_return:
            s, a, r, s_ = get_sample(self.memory.D, self.h.n_step_return)
            self.brain.train_push(s, a, r, s_)
            
            self.R = self.R - self.memory.D[0][2]
            self.memory.removeFirstN(1)    
    
    # possible edge case - if an episode ends in <N steps, the computation is incorrect