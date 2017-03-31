# By Nick Erickson
# Contains class for memory

from collections import deque
import random

# TODO: Implement sum tree for prioritized learning
class Memory: # TODO: use maxlen argument in Deque?
    def __init__(self, max_size):
        self.D = deque()
        self.max_size = max_size
        self.size = 0
        self.total_saved = 0
        self.isFull = False
    
    def add(self, x):
        self.D.append(x)
        if self.size >= self.max_size:
            self.D.popleft()
            self.isFull = True
        else:
            self.size += 1
        self.total_saved += 1
        
    def removeLastN(self, n): # Remove last n instances
        if n > self.size:
            n = self.size
        for i in range(n):
            self.D.pop()
        self.size -= n
        self.total_saved -= n
        self.isFull = False
        
    def sample(self, batch_size):
        return random.sample(self.D, batch_size)
