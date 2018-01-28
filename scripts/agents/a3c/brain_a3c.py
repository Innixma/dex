# By Nick Erickson
# A3C Brain

import time

import numpy as np
import tensorflow as tf
import threading
# Deep Learning Modules
from keras import backend as K

from agents import models
from agents.memory import Memory
from utils import data_aug
from utils.data_utils import load_memory_direct

print('TensorFlow version', tf.__version__)
print(K.learning_phase())


# TODO: Avoid hardcoding memory size
# MEMORY_SIZE = 8
# MEMORY_SIZE = 32000
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, agent, model_func=None):
        self.initialized = False
        self.finalized = False
        self.c = 0
        self.agent = agent
        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim
        self.gamma = self.agent.h.gamma
        self.n_step_return = self.agent.h.memory_size
        self.gamma_n = self.gamma ** self.n_step_return
        self.loss_v = self.agent.h.extra.loss_v
        self.loss_entropy = self.agent.h.extra.loss_entropy
        self.batch = self.agent.h.batch
        self.learning_rate = self.agent.h.learning_rate
        self.brain_memory_size = self.agent.args.hyper.extra.brain_memory_size

        self.env = self.agent.args.env
        self.metrics = self.agent.metrics

        self.brain_memory = Memory(self.brain_memory_size, self.state_dim, self.action_dim)
        if self.agent.args.data:  # Load memory
            s, a, r, s_, t = load_memory_direct('../data/' + self.agent.args.data + '/')
            self.brain_memory.add(s, a, r, s_, t)

        self.NONE_STATE = np.zeros(self.state_dim)

        self.visualization = agent.visualization
        self.model = self.create_model(model_func)

    def init_model(self):
        if self.initialized is True:
            return
        if self.visualization is False:
            self.session = tf.Session()
            K.set_session(self.session)
            K.manual_variable_initialization(True)
            self.graph = self.create_graph(self.model)

            self.session.run(tf.global_variables_initializer())
            self.default_graph = tf.get_default_graph()

        self.initialized = True
        # avoid modifications

    def init_vars(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def finalize_model(self):
        if self.finalized is True:
            return
        self.default_graph.finalize()
        self.finalized = True
        # for layer in self.model.layers:
        #     weights = layer.get_weights()
        #     print(np.sum(np.sum(weights)))
        #     c += 1
        #     print(c)
        #     print(np.sum(layer.get_weights()))

    def create_model(self, model_func=None):
        print(self.state_dim)
        print(self.action_dim)
        if not model_func:
            model_func = models.model_mid_default
        model = models.model_start(self.state_dim, self.action_dim, models.model_top_a3c, model_func, self.visualization)

        model._make_predict_function()  # have to initialize before threading
        print("Finished building the model")
        print(model.summary())
        return model

    def create_graph(self, model):
        batch_size = None # = None
        state_dim = [batch_size] + self.state_dim
        print(state_dim)
        s_t = tf.placeholder(tf.float32, shape=(state_dim))
        a_t = tf.placeholder(tf.float32, shape=(batch_size, self.action_dim))
        r_t = tf.placeholder(tf.float32, shape=(batch_size, 1))  # Discounted Reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-6)  # Negative, larger when action is less likely
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # Pos if better than expected, Neg if bad
        loss_value = self.loss_v * tf.square(advantage)  # Positive # minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-6), axis=1, keep_dims=True)  # Negative Value

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-3)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize, loss_total, log_prob, loss_policy, loss_value, entropy

    def optimize_batch_full(self, reset=1, suppress=1):  # Use for online learning
        if self.brain_memory.is_full != True:
            return

        idx = np.arange(0, self.brain_memory.max_size)

        self.optimize_batch_index(idx, 1, reset, suppress)

    def optimize_batch_full_multithread(self, reset=1, suppress=1):  # Use for online learning
        if self.brain_memory.is_full is False:
            time.sleep(0)  # yield
            return

        idx = np.arange(0, self.brain_memory.max_size)

        self.optimize_batch_index_multithread(idx, 1, reset, suppress)

    def optimize_batch(self, batch_count=1, suppress=0):  # Use for offline learning
        if self.brain_memory.is_full is False:
            time.sleep(0)  # yield
            return

        idx = self.brain_memory.sample(self.batch * batch_count)
        self.optimize_batch_index(idx, batch_count, suppress)

    def optimize_batch_index(self, idx, batch_count=1, reset=0, suppress=0):
        s  = self.brain_memory.s [idx, :]
        a  = self.brain_memory.a [idx, :]
        r  = np.copy(self.brain_memory.r [idx, :])
        s_ = self.brain_memory.s_[idx, :]
        t  = self.brain_memory.t [idx, :]

        if reset == 1:
            self.brain_memory.is_full = False
            self.brain_memory.size = 0

        self.optimize_batch_child(s, a, r, s_, t, batch_count, suppress)

    def optimize_batch_index_multithread(self, idx, batch_count=1, reset=1, suppress=0):
        with self.lock_queue:
            if self.brain_memory.is_full is False:
                return

            s  = np.copy(self.brain_memory.s [idx, :])
            a  = np.copy(self.brain_memory.a [idx, :])
            r  = np.copy(self.brain_memory.r [idx, :])
            s_ = np.copy(self.brain_memory.s_[idx, :])
            t  = np.copy(self.brain_memory.t [idx, :])

            if reset == 1:
                self.brain_memory.is_full = False
                self.brain_memory.size = 0

        self.c += 1
        self.optimize_batch_child(s, a, r, s_, t, batch_count, suppress)

    def optimize_batch_child(self, s, a, r, s_, t, batch_count=1, suppress=0):
        s_t, a_t, r_t, minimize, loss_total, log_prob, loss_policy, loss_value, entropy = self.graph
        for i in range(batch_count):
            start = i * self.batch
            end = (i+1) * self.batch
            r[start:end] = r[start:end] + self.gamma_n * self.predict_v(s_[start:end]) * t[start:end] #  set v to 0 where s_ is terminal state
            _, loss_current, log_current, loss_p_current, loss_v_current, entropy_current = self.session.run([minimize, loss_total, log_prob, loss_policy, loss_value, entropy], feed_dict={s_t: s[start:end], a_t: a[start:end], r_t: r[start:end]})

            # self.metrics.a3c.update(loss_current, log_current, loss_p_current, loss_v_current, entropy_current)

            if i % 10 == 0 and suppress == 0:
                print('\r', 'Learning', '(', i, '/', batch_count, ')', end="")

        if suppress == 0:
            print('\r', 'Learning', '(', batch_count, '/', batch_count, ')')

    def train_augmented(self, s, a, r, s_):
        if self.env.problem == 'Hexagon':
            if s_ is None:
                self.train_push_all_augmented(data_aug.full_augment([[s, a, r, self.NONE_STATE, 0.]]))
            else:
                self.train_push_all_augmented(data_aug.full_augment([[s, a, r, s_, 1.]]))
        else:
            if s_ is None:
                self.train_push_augmented([s, a, r, self.NONE_STATE, 0.])
            else:
                self.train_push_augmented([s, a, r, s_, 1.])

    def train_push_all_augmented(self, frames):
        for frame in frames:
            self.train_push_augmented(frame)

    # TODO: t value is flipped for brain memory and agent memory... should be consistent. Not a bug however.
    def train_push_augmented(self, frame):
        a_cat = np.zeros(self.action_dim)
        a_cat[frame[1]] = 1

        with self.lock_queue:
            if self.brain_memory.is_full is True:
                time.sleep(0)
                return
            self.brain_memory.add_single(frame[0], a_cat, frame[2], frame[3], frame[4])
        # self.train_queue.append([frame[0], a_cat, frame[2], frame[3], frame[4]])

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, _ = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            _, v = self.model.predict(s)
            return v
