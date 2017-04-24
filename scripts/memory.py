# By Nick Erickson
# Contains class for memory

import numpy as np

# TODO: Implement sum tree for prioritized learning
class Memory: # Improved memory class
    def __init__(self, max_size, state_dim, action_dim):
        # state_dim = [96,96,4] example
        
        full_dims = [max_size] + list(state_dim)

        self.s = np.zeros(full_dims, dtype='float16')
        self.a = np.zeros([max_size, action_dim], dtype='int8')
        self.r = np.zeros([max_size, 1], dtype='float64')
        self.s_ = np.zeros(full_dims, dtype='float16')
        self.t = np.zeros([max_size, 1], dtype='int8')
        
        self.max_size = max_size
        self.state_dim = state_dim
        
        self.size = 0
        self.total_saved = 0
        self.isFull = False
        self.curIndex = 0
        
    def add_single(self, s, a, r, s_, t):
        self.s [self.curIndex] = s
        self.a [self.curIndex] = a
        self.r [self.curIndex] = r
        self.s_[self.curIndex] = s_
        self.t [self.curIndex] = t

        self.increment_index_n(1)        

        
    def add(self, s, a, r, s_, t): # Add multiple
        n = s.shape[0]
        
        newIndex = self.curIndex + n
        
        if newIndex >= self.max_size: # Over capacity
            newIndex = newIndex % self.max_size
            split = self.max_size - self.curIndex
            self.s [self.curIndex:self.max_size] = s [:split]
            self.a [self.curIndex:self.max_size] = a [:split]
            self.r [self.curIndex:self.max_size] = r [:split]
            self.s_[self.curIndex:self.max_size] = s_[:split]
            self.t [self.curIndex:self.max_size] = t [:split]

            self.s [:newIndex] = s [split:]
            self.a [:newIndex] = a [split:]
            self.r [:newIndex] = r [split:]
            self.s_[:newIndex] = s_[split:]
            self.t [:newIndex] = t [split:]

        else:
            self.s [self.curIndex:newIndex] = s
            self.a [self.curIndex:newIndex] = a
            self.r [self.curIndex:newIndex] = r
            self.s_[self.curIndex:newIndex] = s_
            self.t [self.curIndex:newIndex] = t
               
        self.increment_index_n(n)


    def increment_index_n(self, n):
        self.curIndex = (self.curIndex + n) % self.max_size
        self.total_saved += n
        if self.isFull == False:
            self.size += n
            if self.size >= self.max_size:
                #print('Brain Memory Filled...')
                self.isFull = True
                self.size = self.max_size
    
    def get_last_n(self, n):
        pointer = (self.curIndex - n) % self.max_size

        return self.s[pointer], self.a[pointer], self.r[pointer], self.s_[pointer], self.t[pointer]
                
    def get_last(self):
        pointer = (self.curIndex - 1) % self.max_size
        
        return self.s[pointer], self.a[pointer], self.r[pointer], self.s_[pointer], self.t[pointer]
                
    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        return idx
        
    def sample_data(self, batch_size):
        idx = self.sample(batch_size)
        
        s  = self.s [idx, :]
        a  = self.a [idx, :]
        r  = np.copy(self.r [idx, :])
        s_ = self.s_[idx, :]
        t  = self.t [idx, :]

        return s, a, r, s_, t
               
    def reset(self):
        self.size = 0
        self.isFull = False
        self.curIndex = 0
