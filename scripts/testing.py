# By Nick Erickson
# Visualizing Data Testing

from data_utils import loadMemory
from agent_random import Agent
from param_const import hex_base_gather
import numpy as np
import data_aug
import time
import memory

state_dim = [64, 64, 2]
action_dim = 3

memory_location = '../data/hex_base_test/'

agent = Agent(hex_base_gather, state_dim, action_dim)
loadMemory(agent, memory_location)

data = list(agent.memory.D)

data = data[0:1]

start = time.time()
for i in range(1000):
    #total_data = data + data_aug.rotate4(data)
    
    #total_data = total_data + data_aug.flip(total_data)
    total_data = data_aug.full_augment(data)
end = time.time()

total_data2 = data_aug.pixel(data)

print(end-start)

state_dims = (64, 64, 2)
max_size = 200000

mem_test = memory.Memory_v2(max_size, state_dims)

s  = np.ones((2000, 64, 64, 2), dtype='float16')
a  = np.ones((2000, 1), dtype='int8')
r  = np.ones((2000, 1), dtype='float64')
s_ = np.ones((2000, 64, 64, 2), dtype='float16')
t  = np.ones((2000, 1), dtype='int8')
for i in range(1000):
    
    
    mem_test.add(s, a, r, s_, t)
    print(np.max(mem_test.sample(100)))

#s  = np.zeros((200000, 96, 96, 4), dtype='float16')
#a  = np.zeros((200000, 1), dtype='int8')
#r  = np.zeros((200000, 1), dtype='float64')
#s_ = np.zeros((200000, 96, 96, 4), dtype='float16')
#t  = np.zeros((200000, 1), dtype='int8') # Boolean?

#print('hi')
#print(s.nbytes)

#print(s.nbytes/1000000)







