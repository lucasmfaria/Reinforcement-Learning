# -*- coding: utf-8 -*-
from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, critic_dropout, learning_rate = 0.001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.critic_dropout = critic_dropout
        self.learning_rate = learning_rate

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        #Add normalization layers
        net_states = layers.BatchNormalization()(states)
        net_actions = layers.BatchNormalization()(actions)
        
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, use_bias=False, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(self.critic_dropout)(net_states)
        net_states = layers.Dense(units=300, use_bias=False, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(self.critic_dropout)(net_states)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400, use_bias=False, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(self.critic_dropout)(net_actions)
        net_actions = layers.Dense(units=300, use_bias=False, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(self.critic_dropout)(net_actions)
        
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        #net = layers.Activation('sigmoid')(net)
        
        # Add more layers to the combined network if needed
        net = layers.Dense(units=300, use_bias=False, kernel_regularizer=regularizers.l2(), bias_regularizer=regularizers.l2())(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dropout(self.critic_dropout)(net)
        
        #Normal distribution initializer for the output layer
        initializer = initializers.RandomNormal(mean=0.0, stddev=0.0015)
        
        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=initializer, bias_initializer=initializer)(net)
        
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)
        
        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)