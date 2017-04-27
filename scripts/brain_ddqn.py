# By Nick Erickson
# DDQN Brain

import models
from keras import backend as K
from keras.optimizers import Adam

def hubert_loss(y_true, y_pred): # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.sqrt(1+K.square(err))-1


# Class concept from Jaromir Janisch, 2016
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
class Brain:
    def __init__(self, agent, modelFunc=None):
        self.agent = agent
        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim

        self.model = self.create_model(modelFunc)
        self.model_ = self.create_model(modelFunc)
        
        self.updateTargetModel()
        
    def create_model(self, modelFunc=None):
        if not modelFunc:
            modelFunc = models.model_mid_default
        model = models.model_start(self.state_dim, self.action_dim, models.model_top_ddqn, modelFunc)
        
        adam = Adam(lr=self.agent.hyper.learning_rate)
        model.compile(loss=hubert_loss,optimizer=adam)
        
        print("Finished building the model")
        print(model.summary())
        return model

    def train(self, states, targets):
        loss = self.model.train_on_batch(states, targets)
        return loss
        
    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        dim = [1] + self.state_dim
        return self.predict(s.reshape(dim), target=target).flatten()
        #return self.predict(s, target=target).flatten()
        
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


