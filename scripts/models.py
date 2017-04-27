# By Nick Erickson
# Model Functions

# TODO: Add lstm to initial layer???
# TODO: Add lstm to final layer???

# TODO: Split into front and back so algorithms can share, much easier

# Deep Learning Modules
from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, MaxPooling2D, BatchNormalization
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
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

def model_start(state_dim, action_dim, model_top, model_mid, visualization=False):
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    model = model_mid(inputs)
    
    model = model_top(inputs, action_dim, model, visualization)
    
    return model
    


def model_mid_default(inputs):

    dense1 = Dense(16, activation='relu')(inputs)

    return dense1
    
def default_model_a3c(state_dim, action_dim, visualization=False):
    l_input = Input( batch_shape=[None] + state_dim )
    l_dense = Dense(16, activation='relu')(l_input)

    out_actions = Dense(action_dim, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    
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
# CNN ddqn, Apr 1
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
# CNN a3c v8, Apr 17
#==============================================================================
def CNN_a3c_LSTM_v8(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv2)
    conv4 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(512, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================


#==============================================================================
# CNN a3c v10, Apr 24 (Downsampled input)
#==============================================================================
def CNN_a3c_v10(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    flatten = Flatten()(conv2)
    dense1 = Dense(256, activation='relu')(flatten)
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================

#==============================================================================
# CNN a3c v9, Apr 21 (Batch Normalization)
#==============================================================================
def CNN_a3_v9(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    #input_norm1 = BatchNormalization()(inputs)
    
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    norm1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(norm1)
    norm2 = BatchNormalization()(conv2)
    
    conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='valid')(norm2)
    norm3 = BatchNormalization()(conv3)
    
    conv4 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(norm3)
    norm4 = BatchNormalization()(conv4)
    
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(norm4)
    norm5 = BatchNormalization()(conv5)
    
    flatten = Flatten()(norm5)
    dense1 = Dense(512, activation='relu')(flatten)
    norm6 = BatchNormalization()(dense1)
    
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    output_actions = Dense(action_dim, activation='softmax')(norm6)
    output_value = Dense(1, activation='linear')(norm6)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================



#==============================================================================
# CNN a3c v8, Apr 21 (Batch Normalization)
#==============================================================================
def CNN_a3c_v8(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    #input_norm1 = BatchNormalization()(inputs)
    
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    norm1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(norm1)
    norm2 = BatchNormalization()(conv2)
    
    conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='valid')(norm2)
    norm3 = BatchNormalization()(conv3)
    
    conv4 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(norm3)
    norm4 = BatchNormalization()(conv4)
    
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(norm4)
    norm5 = BatchNormalization()(conv5)
    
    flatten = Flatten()(norm5)
    dense1 = Dense(512, activation='relu')(flatten)
    norm6 = BatchNormalization()(dense1)
    
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    output_actions = Dense(action_dim, activation='softmax')(norm6)
    output_value = Dense(1, activation='linear')(norm6)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================


#==============================================================================
# CNN a3c v7, Apr 16
#==============================================================================
def CNN_a3c(state_dim, action_dim, visualization=False):
    if visualization == True:
        action_activation = 'linear'
    else:
        action_activation = 'softmax'
    
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv2)
    conv4 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(512, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    
    output_actions = Dense(action_dim, activation=action_activation, name='action')(dense1) # FIX THIS
    output_value = Dense(1, activation='linear', name='value')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================

#==============================================================================
# CNN a3c v6, Apr 11
#==============================================================================
def CNN_a3c_v6(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    conv3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv2)
    conv4 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(512, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    return model
#==============================================================================



#==============================================================================
# CNN a3c v5, Apr 10, TOO SIMPLE, did not learn at all
#==============================================================================
def CNN_a3c_v5(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(4, (4, 4), strides=(2, 2), activation='relu', padding='valid')(inputs)
    conv2 = Conv2D(8, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv1)
    conv3 = Conv2D(8, (4, 4), strides=(2, 2), activation='relu', padding='valid')(conv2)
    flatten = Flatten()(conv3)
    dense1 = Dense(128, activation='relu')(flatten)
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    
    
    return model
#==============================================================================


#==============================================================================
# CNN a3c v4, Apr 10
#==============================================================================
def CNN_a3c_v4(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    conv1 = Conv2D(8, (4, 4), strides=(2, 2), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)
    flatten = Flatten()(conv5)
    dense1 = Dense(256, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    
    
    return model
#==============================================================================


#==============================================================================
# CNN a3c v3, Apr 3, achieved 305s
#==============================================================================
def CNN_a3c_v3(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    maxpool = MaxPooling2D((2, 2))(inputs)
    conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(maxpool)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)
    flatten = Flatten()(conv4)
    dense1 = Dense(256, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    
    
    return model
#==============================================================================

#==============================================================================
# CNN a3c v2, Apr 2
#==============================================================================
def CNN_a3c_v2(state_dim, action_dim):
    
    inputs = Input(shape=state_dim, dtype='float32', name='input')
    
    maxpool = MaxPooling2D((2, 2))(inputs)
    conv1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same')(maxpool)
    conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)
    flatten = Flatten()(conv4)
    dense1 = Dense(256, activation='relu')(flatten)
    #lstm1 = LSTM(256, activation='relu')(dense1)
    
    
    output_actions = Dense(action_dim, activation='softmax')(dense1)
    output_value = Dense(1, activation='linear')(dense1)
    
    model = Model(inputs=[inputs], outputs=[output_actions, output_value])
    
    
    
    return model
#==============================================================================


#==============================================================================
# CNN a3c v1, Apr 1
#==============================================================================
def CNN_a3c_v1(state_dim, action_dim):
    
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


















