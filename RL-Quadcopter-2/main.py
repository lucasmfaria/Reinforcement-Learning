# -*- coding: utf-8 -*-
from task import Task
from agents.agent import Agent
from agents.policy_search import PolicySearch_Agent
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

#Initial conditions
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results
#Goal position
target_pos = np.array([0., 0., 10.])
#Task types
tasks = {
    "1": "takeoff",
    "2": "land"
}

#Task instances
takeoff = Task(task_type=tasks.get("1"))
#takeoff = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos, task_type=tasks.get("1"))
land = Task(task_type=tasks.get("2"))
#land = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos, task_type=tasks.get("2"))

agent = Agent(takeoff)


# Setup
done = False

#Results with the conditions of the quadcopter
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}


num_episodes = 1000


for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = takeoff.step(action)
        agent.step(reward, done)
        state = next_state
        to_write = [takeoff.sim.time] + list(takeoff.sim.pose) + list(takeoff.sim.v) + list(takeoff.sim.angular_v) + list(action)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()


''' Shows the results of the control
plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()
_ = plt.ylim()
'''