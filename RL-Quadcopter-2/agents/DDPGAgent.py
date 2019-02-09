# -*- coding: utf-8 -*-

from actor import Actor
from critic import Critic
from utils.ounoise import OUNoise
from utils.prioritizedreplaybuffer import PrioritizedReplayBuffer
import numpy as np
import csv
from utils.visuals import control_results

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, buffer_size, batch_size, gamma, tau, actor_dropout, critic_dropout, exploration_theta, exploration_sigma, actor_lr, critic_lr):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.actor_dropout = actor_dropout
        self.critic_dropout = critic_dropout
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_dropout, self.actor_lr)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.actor_dropout, self.actor_lr)
        
        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.critic_dropout, self.critic_lr)
        self.critic_target = Critic(self.state_size, self.action_size, self.critic_dropout, self.critic_lr)
        
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        # Noise process
        self.exploration_mu = 5
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(self.buffer_size, self.batch_size)
        
        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters
        
        self.best_score = -np.inf

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        
        self.total_reward = 0.0
        self.count = 0
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        #self.memory.add(self.last_state, action, reward, next_state, done)
        #Generate the parameters in order to calculate the TD error
        next_state_predict = np.reshape(next_state, [-1, self.state_size])
        last_state_predict = np.reshape(self.last_state, [-1, self.state_size])
        action_predict = np.reshape(action, [-1, self.action_size])
        #next_state_action = np.concatenate([next_state, action])
        Q_target_next = self.critic_target.model.predict([next_state_predict, action_predict])[0]
        Q_local = self.critic_local.model.predict([last_state_predict, action_predict])[0]
        
        #Calculate the TD error in order to generate the priority value of the experience
        td_error = reward + self.gamma*Q_target_next - Q_local
        
        #Normalize the TD error with TANH as advised by Google's DeepMind paper "Prioritized Experience Replay": https://arxiv.org/pdf/1511.05952.pdf
        #td_error = math.tanh(td_error[0])
        
        self.memory.add(self.last_state, action, reward, next_state, done, abs(td_error[0]))
        
        self.total_reward += reward
        self.count += 1
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences, idx_sample, is_weights = self.memory.sample_priority()
            self.learn(experiences, idx_sample, is_weights)
        
        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, test=False):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        if test == False:
            return list(action + self.noise.sample())  # add some noise for exploration
        else:
            return list(action)

    def learn(self, experiences, idx_sample, is_weights):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        is_weights = is_weights.reshape(-1,1)
        
        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones) * is_weights
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
        #Generate the new TD error value and update the priority value within the Replay Buffer
        td_error = rewards + self.gamma*Q_targets_next * (1 - dones) - Q_targets
        
        #Normalize the TD error with TANH as advised by Google's DeepMind paper "Prioritized Experience Replay": https://arxiv.org/pdf/1511.05952.pdf
        #td_error = np.tanh(td_error)
        
        self.memory.update_priority(idx=idx_sample, error=td_error)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def test_control(self, file_output='data.txt'):
        state = self.reset_episode()
        done = False
        #Results with the conditions of the quadcopter
        labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
                  'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
                  'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
        results = {x : [] for x in labels}
        
        # Run the simulation, and save the results.
        with open(file_output, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            while True:
                action = self.act(state, test=True)
                #action = self.act(state, test=False)
                next_state, reward, done = self.task.step(action)
                state = next_state
                to_write = [self.task.sim.time] + list(self.task.sim.pose) + list(self.task.sim.v) + list(self.task.sim.angular_v) + list(action)
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)
                if done:
                    break
        #Shows the results of the control
        control_results(results)

    #Useful for testing
    def update_score(self):
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score