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
        
        load_weights(self)
        self.brain.updateTargetModel()

    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final and self.memory.total_saved > self.h.observe:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore
        
    def update_agent(self):
        if self.update_iterator >= self.h.update_rate:
            self.update_iterator -= self.h.update_rate
            print('Updating Target Network')
            self.brain.updateTargetModel()
            
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            return np.argmax(self.brain.predictOne(s))
    
    def observe(self, sample):
        self.memory.add(sample)
        self.update_epsilon()
        self.save_iterator += 1
        #self.update_iterator += 1
        
    def replay(self, debug=True):
        self.replay_count += 1
        self.update_iterator += 1
        Q_sa_total = 0
        
        batch = self.memory.sample(self.h.batch)
        batchLen = len(batch)
        
        states = np.array([x[0] for x in batch])
        states_ = np.array([x[3] for x in batch])
        
        targets = self.brain.predict(states)
        targets_ = self.brain.predict(states_, target=False) # Target Network!              
        pTarget_ = self.brain.predict(states_, target=True)                    
        Q_size = batchLen
        
        # TODO: Prioritized experience replay
        for i in range(0, batchLen):
            action_t = batch[i][1]
            reward_t = batch[i][2]
            terminal = batch[i][4]
            
            if terminal:
                Q_size -= 1
                targets[i, action_t] = reward_t
            else:
                Q_sa_total += np.max(targets_[i])
                #targets[i, action_t] = reward_t + self.h.gamma * np.max(targets_[i]) # Full DQN (Worse than double DQN)
                targets[i, action_t] = reward_t + self.h.gamma * pTarget_[i][np.argmax(targets_[i])]  # double DQN

        #loss = self.brain.model.train_on_batch(states, targets) # Maybe do fit in future
        loss = self.brain.train(states, targets)
        if Q_size == 0:
            Q_size = 1
        Q_sa_total = Q_sa_total/Q_size
        
        if debug:
            print("\tQ %.2f" % Q_sa_total, "/ L %.2f" % loss)
        
        if self.replay_count % 100 == 0:
            self.metrics.Q.append(Q_sa_total) # TODO: Save these better
            self.metrics.loss.append(loss)

        self.update_agent()
            