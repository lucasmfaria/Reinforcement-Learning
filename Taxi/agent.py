import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.1, gamma = 0.9):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #self.epsilon = 0.9
        self.alpha = alpha
        self.i_episode = 0
        self.gamma = gamma

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
    # updates the action-value function estimate using the most recent time step 
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
    
    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        #epsilon = 1.0 / i_episode
        #testing other decay rates
        if i_episode<=12000:
            epsilon = -(0.9/12000)*i_episode+1
        else:
            epsilon = 0.001
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state, debug = False):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.i_episode += 1
        policy_s = self.epsilon_greedy_probs(self.Q[state], self.i_episode)
        action = np.random.choice(self.nA, p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward, self.alpha, self.gamma)
    
    def select_greedy_action(self, state):
        #policy_sarsamax = np.array([np.argmax(self.Q[key]) if key in self.Q else -1 for key in np.arange(500)])
        return np.argmax(self.Q[state])