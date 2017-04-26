# By Nick Erickson
# DDQN Brain

from models import default_model


# Class concept from Jaromir Janisch, 2016
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
class Brain:
    def __init__(self, state_dim, action_dim, modelFunc=None):
        self.state_dim = state_dim
        print(state_dim)
        print(action_dim)
        self.action_dim = action_dim

        self.model = self.create_model(modelFunc)
        self.model_ = self.create_model(modelFunc)
        
        self.updateTargetModel()
        
    def create_model(self, modelFunc=None):
        if modelFunc:
            model = modelFunc(self.state_dim, self.action_dim)
        else:
            model = default_model(self.state_dim, self.action_dim)
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


