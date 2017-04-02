# By Nick Erickson
# A3C Brain

# Deep Learning Modules
from keras import backend as K
from keras.models import Model, Input
from keras.models import model_from_json
from keras.layers.core import Dense
from keras.optimizers import SGD , Adam , RMSprop

import tensorflow as tf
from models import default_model

import numpy as np

# Class concept from Jaromir Janisch, 2017
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
class Brain:
    train_queue = [ [], [], [], [], [] ]    # s, a, r, s', s' terminal mask

    def __init__(self, state_dim, action_dim, hyper, modelFunc=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = hyper.gamma
        self.n_step_return = hyper.memory_size
        self.gamma_n = self.gamma ** self.n_step_return
        self.loss_v = hyper.extra.loss_v
        self.loss_entropy = hyper.extra.loss_entropy
        self.batch = hyper.batch
        self.learning_rate = hyper.learning_rate
        
        self.NONE_STATE = np.zeros(state_dim)
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self.create_model(modelFunc)
        self.graph = self.create_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()    # avoid modifications

    def create_model(self, modelFunc=None):
        print(self.state_dim)
        print(self.action_dim)
        if modelFunc:
            model = modelFunc(self.state_dim, self.action_dim)
        else:
            l_input = Input( batch_shape=(None, self.state_dim[0]) )
            l_dense = Dense(16, activation='relu')(l_input)
    
            out_actions = Dense(self.action_dim, activation='softmax')(l_dense)
            out_value   = Dense(1, activation='linear')(l_dense)
    
            model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function() # have to initialize before threading
        print("Finished building the model")
        print(model.summary())
        return model
        
    def create_graph(self, model):
        #print(self.state_dim)
        #print(self.state_dim[0])
        zzz = [None] + self.state_dim
        print(zzz)
        s_t = tf.placeholder(tf.float32, shape=(zzz))
        a_t = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
        
        p, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)                                    # maximize policy
        loss_value  = self.loss_v * tf.square(advantage)                                                # minimize value error
        entropy = self.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)    # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < self.batch:
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [ [], [], [], [], [] ]

        a_cats = []
        for a_ in a:
            a_cat = np.zeros(self.action_dim)
            a_cat[a_] = 1
            a_cats.append(a_cat)
        #print(np.array(s[0]).shape)
        #print(np.array(s).shape)
        #s = np.vstack(s)
        
        s = np.array(s)
        s_ = np.array(s_)
        #print(s.shape)
        a = np.vstack(a_cats)
        r = np.vstack(r)
        #s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        v = self.predict_v(s_)
        r = r + self.gamma_n * v * s_mask    # set v to 0 where s_ is terminal state
        
        s_t, a_t, r_t, minimize = self.graph

        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)
        if s_ is None:
            self.train_queue[3].append(self.NONE_STATE)
            self.train_queue[4].append(0.)
        else:    
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

        self.optimize()
            
    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)        
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v