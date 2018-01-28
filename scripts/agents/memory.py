# By Nick Erickson
# Contains class for memory

import numpy as np


# TODO: Implement sum tree for prioritized learning
class Memory:  # Improved memory class
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
        self.is_full = False
        self.cur_index = 0

    def add_single(self, s, a, r, s_, t):
        self.s [self.cur_index] = s
        self.a [self.cur_index] = a
        self.r [self.cur_index] = r
        self.s_[self.cur_index] = s_
        self.t [self.cur_index] = t

        self.increment_index_n(1)

    def add(self, s, a, r, s_, t):  # Add multiple
        n = s.shape[0]
        if n > self.max_size:
            self.cur_index = 0
        new_index = self.cur_index + n

        if new_index >= self.max_size: # Over capacity
            new_index = new_index % self.max_size
            split = self.max_size - self.cur_index
            self.s [self.cur_index:self.max_size] = s [:split]
            self.a [self.cur_index:self.max_size] = a [:split]
            self.r [self.cur_index:self.max_size] = r [:split]
            self.s_[self.cur_index:self.max_size] = s_[:split]
            self.t [self.cur_index:self.max_size] = t [:split]

            if n <= self.max_size:
                self.s [:new_index] = s [split:]
                self.a [:new_index] = a [split:]
                self.r [:new_index] = r [split:]
                self.s_[:new_index] = s_[split:]
                self.t [:new_index] = t [split:]

        else:
            self.s [self.cur_index:new_index] = s
            self.a [self.cur_index:new_index] = a
            self.r [self.cur_index:new_index] = r
            self.s_[self.cur_index:new_index] = s_
            self.t [self.cur_index:new_index] = t

        if n > self.max_size:
            n = self.max_size
        self.increment_index_n(n)

    def increment_index_n(self, n):
        self.cur_index = (self.cur_index + n) % self.max_size
        self.total_saved += n
        if self.is_full == False:
            self.size += n
            if self.size >= self.max_size:
                # print('Brain Memory Filled...')
                self.is_full = True
                self.size = self.max_size

    def get_last_n(self, n):
        pointer = (self.cur_index - n) % self.max_size

        return self.s[pointer], self.a[pointer], self.r[pointer], self.s_[pointer], self.t[pointer]

    def get_last(self):
        pointer = (self.cur_index - 1) % self.max_size

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
        self.is_full = False
        self.cur_index = 0
