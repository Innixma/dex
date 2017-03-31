# By Nick Erickson
# A3C Agent

import numpy as np
import globs as G
import os
from memory import Memory
from data_utils import Metrics
from brain_a3c import Brain
import json
import random
import graphHelper

class Agent:
    def __init__(self, hyperparams, args, state_dim, action_dim, modelFunc=None):
        self.h = hyperparams
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
        
        self.load_weights()
        self.brain.updateTargetModel()
        
    def load_weights(self): # TODO: Update this function
        if self.args.directory == 'default':
            self.args.directory = G.CUR_FOLDER

        results_location = G.RESULT_FOLDER_FULL + '/' + self.args.directory
        data_location = G.DATA_FOLDER_FULL + '/' + self.args.directory
        os.makedirs(results_location,exist_ok=True) # Generates results folder
        os.makedirs(data_location,exist_ok=True) # Generates data folder
        self.results_location = results_location + '/'
        self.data_location = data_location + '/'
        
        if self.args.mode == 'run':
            self.h.observe = 999999999    # Never train
            self.epsilon = 0
            print ("Now we load weight from " + self.results_location + 'model.h5')
            self.brain.model.load_weights(self.results_location + 'model.h5')

            print ("Weights loaded successfully")
        elif self.args.mode == 'train_old': # Continue training old network
            self.h.observe = self.h.observe
            self.epsilon = self.h.epsilon_init
            print ("Now we load weight from " + self.results_location + 'model.h5')
            self.brain.model.load_weights(self.results_location + 'model.h5')

            print ("Weights loaded successfully, training")
        else: # Train new
            print('Training new network!')
            self.h.observe = self.h.observe
            self.epsilon = self.h.epsilon_init
     
    def save_weights(self):
        print("Saving Model...")
        self.brain.model.save_weights(self.results_location + 'model.h5', overwrite=True)
        with open(self.results_location + 'model.json', "w") as outfile:
            json.dump(self.brain.model.to_json(), outfile)
            
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
            #state_t = batch[i][0]
            action_t = batch[i][1]
            reward_t = batch[i][2]
            #state_t1 = batch[i][3]
            terminal = batch[i][4]
            
            if terminal:
                Q_size -= 1
                targets[i, action_t] = reward_t
            else:
                Q_sa_total += np.max(targets_[i])
                #targets[i, action_t] = reward_t + self.h.gamma * np.max(targets_[i]) # Full DQN (Worse than double DQN)
                targets[i, action_t] = reward_t + self.h.gamma * pTarget_[i][np.argmax(targets_[i])]  # double DQN

        loss = self.brain.model.train_on_batch(states, targets) # Maybe do fit in future
        if Q_size == 0:
            Q_size = 1
        Q_sa_total = Q_sa_total/Q_size
        
        if debug:
            print("\tQ %.2f" % Q_sa_total, "/ L %.2f" % loss)
        
        if self.replay_count % 100 == 0:
            self.metrics.Q.append(Q_sa_total) # TODO: Save these better
            self.metrics.loss.append(loss)
            #self.save_metrics_training() # TODO: move this
            
    def display_metrics(self, frame, useRate):
        if np.sum(useRate) != 0:
            useRate = useRate/np.sum(useRate)
        framerate = frame/self.metrics.survival_times[-1]
        print('R' + str(self.run_count) + ': ' + "%.2f" % self.metrics.survival_times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate], end='')
        print(' Mean: %.2f' % np.mean(self.metrics.survival_times), 'Last 10: %.2f' % self.metrics.survival_times_last_10[-1], 'Max: %.2f' % np.max(self.metrics.survival_times), "TS", self.memory.total_saved, "E %.2f" % self.epsilon)

    def save_metrics_training(self):
        graphHelper.graphSimple([np.arange(len(self.metrics.Q))], [self.metrics.Q], ['Q Value'], 'Q Value', 'Q Value', 'Replay (10^2)', savefigName=self.results_location + 'Q_graph')
        graphHelper.graphSimple([np.arange(len(self.metrics.loss))], [self.metrics.loss], ['Loss'], 'Loss', 'Loss', 'Replay (10^2)', savefigName=self.results_location + 'Loss_graph')        
        
    def save_metrics(self):
        # TODO: Remove these logs, just export the data directly
        graphHelper.graphSimple([np.arange(self.run_count+1),np.arange(self.run_count+1),np.arange(self.run_count+1)], [self.metrics.survival_times, self.metrics.survival_times_last_10, self.metrics.survival_times_full_mean], ['DQN', 'DQN_Last_10_Mean', 'DQN_Full_Mean'], 'DQN', 'Time(s)', 'Run', savefigName=self.results_location + 'DQN_graph')
