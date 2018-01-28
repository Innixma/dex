# By Nick Erickson
# DDQN Agent

import random

import numpy as np

from agents.ddqn.brain_ddqn import Brain
from agents.memory import Memory
from agents.metrics import Metrics
from utils.data_utils import load_weights


class Agent:
    def __init__(self, args, state_dim, action_dim, modelFunc=None):
        print(state_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h = args.hyper
        self.metrics = Metrics()
        self.memory = Memory(self.h.memory_size, self.state_dim, 1)
        self.brain = Brain(self, modelFunc)
        self.args = args
        self.epsilon = self.h.epsilon_init

        self.run_count = -1
        self.replay_count = -1
        self.save_iterator = -1
        self.update_iterator = -1
        self.mode = 'observe'

        load_weights(self)
        self.brain.update_target_model()

    def update_epsilon(self):
        if self.epsilon > self.h.epsilon_final and self.memory.total_saved > self.h.extra.observe:
            self.epsilon -= (self.h.epsilon_init - self.h.epsilon_final) / self.h.explore

    def update_agent(self):
        if self.update_iterator >= self.h.extra.update_rate:
            self.update_iterator -= self.h.extra.update_rate
            print('Updating Target Network')
            self.brain.update_target_model()

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            return np.argmax(self.brain.predict_one(s))

    def observe(self, s, a, r, s_, t):
        self.memory.add_single(s, a, r, s_, t)
        self.update_epsilon()
        self.save_iterator += 1

    def replay(self, debug=True):
        self.replay_count += 1
        self.update_iterator += 1
        Q_sa_total = 0

        s, a, r, s_, t = self.memory.sample_data(self.h.batch)

        targets = self.brain.predict(s)
        targets_ = self.brain.predict(s_, target=False)  # Target Network!
        pTarget_ = self.brain.predict(s_, target=True)
        Q_size = self.h.batch - np.sum(t)
        if Q_size == 0:
            Q_size = 1
        # TODO: Prioritized experience replay
        for i in range(0, self.h.batch):
            if t[i]:
                targets[i, a[i]] = r[i]
            else:
                Q_sa_total += np.max(targets_[i])
                targets[i, a[i]] = r[i] + self.h.gamma * pTarget_[i][np.argmax(targets_[i])]  # double DQN

        loss = self.brain.train(s, targets)
        Q_sa_total = Q_sa_total/Q_size

        if debug:
            print("\tQ %.2f" % Q_sa_total, "/ L %.2f" % loss)

        if self.replay_count % 100 == 0:
            self.metrics.Q.append(Q_sa_total) # TODO: Save these better
            self.metrics.loss.append(loss)

        self.update_agent()
