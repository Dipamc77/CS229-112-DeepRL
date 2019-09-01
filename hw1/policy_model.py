from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def policy_model(observation_space, action_space, learning_rate=1e-2, l2_reg=0, batchnorm=False):
    """ 
    Policy model for Homework 1 of 2018 CS229-112 Deep Reinforcement Learning
    The expert policies provided all have three layer fully connected networks
    The architecture is as follows:
        Layer 1: 64 Hidden units with tanh activation
        Layer 2: 64 Hidden units with tanh activation
        Layer 3: Ac Hidden units without any activation
            Here Ac is the number of discrete actions
    The policy is always GaussianPolicy hence loss should be MSE
    Trainining is done with Adam Optimizer
    """
    in_shape = observation_space.shape
    ac_shape = action_space.shape
    
    l2_reg = regularizers.l2(l2_reg)
    model = Sequential()    
    model.add(Dense(64, input_dim=in_shape[0], kernel_regularizer=l2_reg))
    if batchnorm == True:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(64, input_dim=(64,), kernel_regularizer=l2_reg))
    if batchnorm == True:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(ac_shape[0], input_dim=(64,), kernel_regularizer=l2_reg))

    adam = Adam(lr=learning_rate)
    model.compile(optimizer='adam', loss='mse')     
    
    return model