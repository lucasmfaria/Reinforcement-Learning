# -*- coding: utf-8 -*-

from agents.DDPGAgent import DDPG
from task import Task
import numpy as np

#Initial conditions
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 0., 0., 0., 0.])  # initial pose
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
#takeoff = Task(init_pose=init_pose, target_pos=target_pos)
takeoff = Task(init_pose=init_pose, init_velocities=init_velocities, init_angle_velocities=init_angle_velocities, runtime=runtime, target_pos=target_pos)
#land = Task(task_type=tasks.get("2"))
#land = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos, task_type=tasks.get("2"))

agent = DDPG(takeoff)

state = agent.reset_episode() # start a new episode
print("State",state)
print("Task pose",takeoff.sim.pose)
action = agent.act(state)
print("Action",action)
next_state, reward, done = agent.task.step(action)
print("Task pose",takeoff.sim.pose)
print("Task pose",agent.task.sim.pose)