# -*- coding: utf-8 -*-

import random
from collections import namedtuple, deque
from math import pow
import numpy as np

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "p"])
        self.e = 0.0000001
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.00005

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        p = self.get_priority(error)
        e = self.experience(state, action, reward, next_state, done, p)
        self.memory.append(e)
        sorted_memory_p = deque(sorted(self.memory, key=lambda m: m.p), maxlen=self.buffer_size)
        self.memory = sorted_memory_p

    def sample_priority(self, batch_size=64):
        """Sample a mini batch with sample probability based on the priority value.
        Return the mini batch, the indexes of each experience within the mini batch, and the Importance Weight of each experience"""
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        priorities = []
        for m in self.memory:
            priorities.append(m.p)
        priorities = np.array(priorities)
        idx = np.array(range(len(priorities)))
        sampling_probabilities = priorities/np.sum(priorities)
        idx_sample = np.random.choice(a=idx, size=self.batch_size, replace=False, p=sampling_probabilities)
        j = 0
        mini_batch = deque(maxlen=self.batch_size)
        #Initialize the Importance Weight:
        is_weights = np.array(range(self.batch_size))
        for j in idx_sample:
            mini_batch.append(self.memory[j])
        is_weights = np.power(len(self.memory) * sampling_probabilities[idx_sample], -self.beta)
        max_weight = is_weights.max()
        is_weights = np.divide(is_weights,max_weight)
        return mini_batch, idx_sample, is_weights

    def random_sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def get_priority(self, error):
        """Return the priority value based on the given TD error"""
        return pow((error + self.e), self.a)

    def update_priority(self, idx, error):
        """Update the priority value within the memory based on the new TD error"""
        for i,e in zip(idx, error):
            self.memory[i] = self.memory[i]._replace(p=self.get_priority(abs(e)))
