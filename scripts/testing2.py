# By Nick Erickson
# Visualizing Data Testing

from data_utils import loadMemory
from agent_random import Agent
from param_const import hex_base_gather
import numpy as np
import data_aug
import time
import memory
import sys


print('hello', end="")
sys.stdout.flush()
time.sleep(2)
print('\rhey', end="")
sys.stdout.flush()

for i in range(1000):
    if i % 10 == 0:
        print('\r', 'Learning', '(', str(i), '/', str(1000), ')', end="")
        sys.stdout.flush()
    time.sleep(0.01)
print('\r', 'Learning Complete                                           ')