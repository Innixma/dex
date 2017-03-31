# By Nick Erickson
# Model Functions

# Deep Learning Modules
from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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
def buildmodel_CNN_v3(state_dim, action_dim):
    
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=state_dim))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
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