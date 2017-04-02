# By Nick Erickson
# Model Functions

# Deep Learning Modules
from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD , Adam , RMSprop

def hubert_loss(y_true, y_pred): # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.sqrt(1+K.square(err))-1

def default_model(state_dim, action_dim):
    model = Sequential()

    model.add(Dense(output_dim=64, activation='relu', input_shape=state_dim))
    model.add(Dense(output_dim=action_dim, activation='linear'))

    opt = RMSprop(lr=0.00025)
    model.compile(loss=hubert_loss, optimizer=opt)
    #model.compile(loss='mse', optimizer=opt)
    return model

#==============================================================================
# CNN model structure (Base v3), Mar 18
#==============================================================================
def buildmodel_CNN_v3_orig(state_dim, action_dim):
    
    model = Sequential()
    model.add(Conv2D(32, (8, 8), input_shape=state_dim, strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_dim))
    model.add(Activation('linear'))
    

    adam = Adam(lr=0.00025)
    model.compile(loss=hubert_loss,optimizer=adam) # Maybe try huber or mae??

    return model
#==============================================================================

#==============================================================================
# CNN a3c, Apr 1
#==============================================================================
def buildmodel_CNN_v3(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    flatten = Flatten()(conv3)
    dense1 = Dense(512, activation='relu')(flatten)
    output = Dense(action_dim, activation='linear')(dense1)
    
    model = Model(inputs=inputs, outputs=output)
    
    adam = Adam(lr=0.00025)
    model.compile(loss=hubert_loss,optimizer=adam) # Maybe try huber or mae??
    
    return model
#==============================================================================
 
#==============================================================================
# CNN a3c, Apr 1
#==============================================================================
def CNN_a3c(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(inputs)
    conv2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(conv2)
    flatten = Flatten()(conv3)
    dense1 = Dense(256, activation='relu')(flatten)
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    
    
    return model
#==============================================================================




















