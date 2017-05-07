# By Nick Erickson
# Model Functions

# TODO: Add lstm to initial layer???
# TODO: Add lstm to final layer???

# TODO: Split into front and back so algorithms can share, much easier

# Deep Learning Modules
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, MaxPooling2D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD , Adam , RMSprop

def hubert_loss(y_true, y_pred): # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.sqrt(1+K.square(err))-1

def model_top_a3c(inputs, action_dim, model, visualization=False):
    if visualization == True:
        action_activation = 'linear'
    else:
        action_activation = 'softmax'
    
    output_actions = Dense(action_dim, activation=action_activation, name='action')(model)
    output_value = Dense(1, activation='linear', name='value')(model)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])

    return model

def model_top_ddqn(inputs, action_dim, model, visualization=False):
    
    output_actions = Dense(action_dim, activation='linear', name='action')(model)
    
    model = Model(inputs=[inputs], outputs=[output_actions])
    
    return model
    
def model_mid_cnn(inputs):
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv2)
    conv4 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(512, activation='relu')(flatten)
    
    return dense1

def model_mid_cnn_42x42(inputs):
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv2)
    flatten = Flatten()(conv3)
    dense1 = Dense(512, activation='relu')(flatten)
    
    return dense1


def model_mid_cnn_42x42_pool(inputs):
    conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(64, (4, 4), strides=(1, 1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    flatten = Flatten()(pool2)
    dense1 = Dense(512, activation='relu')(flatten)
    
    return dense1
        
    
def model_mid_atari(inputs):
    conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
    flatten = Flatten()(conv2)
    dense1 = Dense(256, activation='relu')(flatten)
    
    return dense1

def model_start(state_dim, action_dim, model_top, model_mid, visualization=False):
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    model = model_mid(inputs)
    
    model = model_top(inputs, action_dim, model, visualization)
    
    return model
    


def model_mid_default(inputs):

    dense1 = Dense(16, activation='relu')(inputs)

    return dense1
