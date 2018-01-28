# By Nick Erickson
# DDQN Brain

from keras import backend as K
from keras.optimizers import Adam

from agents import models


def hubert_loss(y_true, y_pred):  # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.sqrt(1+K.square(err))-1


class Brain:
    def __init__(self, agent, model_func=None):
        self.agent = agent
        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim

        self.model = self.create_model(model_func)
        self.model_ = self.create_model(model_func)

        self.update_target_model()

    def create_model(self, model_func=None):
        if not model_func:
            model_func = models.model_mid_default
        model = models.model_start(self.state_dim, self.action_dim, models.model_top_ddqn, model_func)

        adam = Adam(lr=self.agent.h.learning_rate)
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

    def predict_one(self, s, target=False):
        dim = [1] + self.state_dim
        return self.predict(s.reshape(dim), target=target).flatten()
        # return self.predict(s, target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())


