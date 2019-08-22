import tensorflow as tf
import numpy as np

class Policy:
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
    def __init__(self, sess, observation_space, action_space, scope):
        """
        sess - The tensorflow session
        observation_space - the observation space of the gym environment
        action_space - the action space of the gym environment
        Note: The learning rate is not parameterized
        """
        self.sess = sess
        in_shape = observation_space.shape
        ac_shape = action_space.shape
        
        self.input_ph = tf.placeholder(tf.float32, shape=(None, *in_shape))
        self.target_ph = tf.placeholder(tf.float32, shape=(None, *ac_shape))
        
        with tf.variable_scope(scope):
            x = tf.layers.dense(self.input_ph, 64, activation='tanh')
            x = tf.layers.dense(x, 64, activation='tanh')
            self.actions = tf.layers.dense(x, ac_shape[0], activation=None)
        
        self.loss = tf.reduce_mean((self.target_ph - self.actions)**2)
        
        train_vars = tf.trainable_variables(scope)
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.loss, var_list=train_vars)
        
    def train(self, inputs, targets):
        """ Train the model on a minibatch """
        return self.sess.run([self.loss, self.train_op], 
                             feed_dict={self.input_ph: inputs,
                                        self.target_ph: targets})
    
    def predict(self, inputs):
        """ Get policy action predictions on a minibatch """
        if len(inputs.shape) == 1:
            inputs = inputs[np.newaxis, :]
        return self.sess.run(self.actions, 
                             feed_dict={self.input_ph: inputs})