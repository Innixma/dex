# By Nick Erickson
# Visualizing Data Testing

from data_utils import loadMemory
from agent_random import Agent
from param_const import hex_base_gather
import numpy as np
import data_aug
import time

state_dim = [64, 64, 2]
action_dim = 3

memory_location = '../data/hex_base_test/'

agent = Agent(hex_base_gather, state_dim, action_dim)
loadMemory(agent, memory_location)

data = list(agent.memory.D)

processed = []
for d in data:
    processed.append(d[0])
processed = np.array(processed)

processed = processed[:,:,:,0]

processed_1 = processed[24]

processed_2 = np.rot90(processed_1)

processed_3 = np.rot90(processed_2)

processed_4 = np.rot90(processed_3)

data = data[0:1]

start = time.time()
for i in range(1000):
    #total_data = data + data_aug.rotate4(data)
    
    #total_data = total_data + data_aug.flip(total_data)
    total_data = data_aug.full_augment(data)
end = time.time()

print(end-start)









