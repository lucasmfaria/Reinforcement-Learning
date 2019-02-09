# -*- coding: utf-8 -*-
#%load_ext autoreload
#%autoreload 2
from task import Task
from agents.DDPGAgent import DDPG
import numpy as np
import sys
from collections import deque
from utils.visuals import check_rewards

#Initial conditions
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data_spyder.txt'                         # file name for saved results
file_output_training = 'test_data_during_training_spyder.txt'      # file name for saved results during training... this gives a visibility during training episodes
#Goal position
target_pos = np.array([0., 0., 20.])
num_episodes = 1500
#Window to calculate reward/score average: 10%
window = int(num_episodes*0.1)

action_repeat=3
buffer_size = 1000000
batch_size = 64
gamma = 0.99
tau = 0.001
actor_dropout = 0.25
critic_dropout = 0.25
exploration_theta = 0.2
exploration_sigma = 0.3
actor_lr = 0.001
critic_lr = 0.001

task = Task(action_repeat=action_repeat, init_pose=init_pose, init_velocities=init_velocities, init_angle_velocities=init_angle_velocities, runtime=runtime, target_pos=target_pos)
agent = DDPG(task=task, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau, actor_dropout=actor_dropout, critic_dropout=critic_dropout, exploration_theta=exploration_theta, exploration_sigma=exploration_sigma, actor_lr=actor_lr, critic_lr=critic_lr)

percentage = 10
test_training = np.arange(0,10,np.divide(percentage,10))
test_training = np.divide(test_training, 10)
# Setup
done = False

# initialize average rewards
avg_rewards = []
for i in range(window):
    avg_rewards.append(np.NaN)
# initialize best average reward
best_avg_reward = -np.inf
# initialize monitor for most recent rewards
samp_rewards = deque(maxlen=window)
rewards = deque(maxlen=num_episodes)
avg_reward=0

for i_episode in range(1, num_episodes+1):
    samp_reward = 0
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = agent.task.step(action)
        samp_reward += reward
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            samp_rewards.append(samp_reward)
            rewards.append(samp_reward)
            break
    if (i_episode >= window):
            # get average reward from last "window" episodes
            avg_reward = np.mean(samp_rewards)
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
    # shows the agent's performance after "percentage"*num_episodes
    if np.divide(i_episode,num_episodes) in (test_training) or i_episode == 1:
        agent.test_control(file_output=file_output_training)
    print("\rEpisode = {:4d}, Total Reward = {:7.3f}, Average Reward = {:7.3f} (best = {:7.3f})".format(i_episode, samp_reward, avg_reward, best_avg_reward), end="")
    sys.stdout.flush()

#Using the learned policy to run an episode
agent.test_control(file_output=file_output)
#Show the rewards over the episodes
check_rewards(rewards=rewards, avg_rewards=avg_rewards)


#from keras.utils.vis_utils import plot_model
#plot_model(agent.actor_local.model, to_file='actor.png', show_shapes=True, show_layer_names=True)
#plot_model(agent.critic_local.model, to_file='critic.png', show_shapes=True, show_layer_names=True)
