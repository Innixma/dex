# By Nick Erickson
# Visualizing Data Testing

import numpy as np
from param_const import hex_base_a3c

from agents import models
from agents.a3c.agent_a3c import Agent as Agent_a3c
from utils.data_utils import load_memory_v2


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

state_dim = [96, 96, 2]
action_dim = 3

memory_location = '../data/hex_hexreal_test/'

#agent = Agent(hex_base_gather, state_dim, action_dim)
agent = Agent_a3c(hex_base_a3c, state_dim, action_dim, modelFunc=models.CNN_a3c)
load_memory_v2(agent, memory_location)

data = agent.brain.brain_memory.s

d = data.reshape(15000, 96*96*2)
dd = zca_whitening_matrix(d[:1000])

xZCAMatrix = np.dot(dd, d[:1000])
xZCAMatrix = xZCAMatrix.reshape(1000, 96, 96, 2)
#dd = preprocessing.normalize(d)

#dd = dd.reshape(15000, 96, 96, 2)

#data2 = dd[4000]
#dataMean = np.mean(data, dtype='float64', axis=(0))

#data3 = data2 - dataMean


#data4 = transf.downscale_local_mean(data3, (2,2,1))

#start = time.time()
#for i in range(1000):
    #total_data = data + data_aug.rotate4(data)

    #total_data = total_data + data_aug.flip(total_data)
    #total_data = data_aug.full_augment(data)
#end = time.time()

#total_data2 = data_aug.pixel(data)

#print(end-start)

#max_size = 200000

#mem_test = memory.Memory_v2(max_size, state_dims, action_dim)

#s  = np.ones((2000, 64, 64, 2), dtype='float16')
#a  = np.ones((2000, 1), dtype='int8')
#r  = np.ones((2000, 1), dtype='float64')
#s_ = np.ones((2000, 64, 64, 2), dtype='float16')
#t  = np.ones((2000, 1), dtype='int8')
#for i in range(1000):


#    mem_test.add(s, a, r, s_, t)
#    print(np.max(mem_test.sample(100)))

#s  = np.zeros((200000, 96, 96, 4), dtype='float16')
#a  = np.zeros((200000, 1), dtype='int8')
#r  = np.zeros((200000, 1), dtype='float64')
#s_ = np.zeros((200000, 96, 96, 4), dtype='float16')
#t  = np.zeros((200000, 1), dtype='int8') # Boolean?

#print('hi')
#print(s.nbytes)

#print(s.nbytes/1000000)







