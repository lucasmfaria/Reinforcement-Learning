from agent import Agent
from monitor import interact
from monitor import run_estimated_policy
import gym
import numpy as np

env = gym.make('Taxi-v2')
#initializing the agent
agent = Agent(alpha = 0.1, gamma = 0.9)
#interacts with the environtment during num_episodes
avg_rewards, best_avg_reward = interact(env, agent, num_episodes = 20000)
#shows the estimated optimal policy undefinitely rendering the environment
while True:
    run_estimated_policy(agent, env)