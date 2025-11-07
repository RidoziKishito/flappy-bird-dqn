import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    Parameters:
        capacity (int): Maximum number of experiences to store. Default: 100000
        
    Methods:
        push: Add a new experience to buffer
        sample: Randomly sample a batch of experiences
        __len__: Return current buffer size
        
    Example:
        >>> buffer = ReplayBuffer(capacity=1000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> states, actions, rewards, next_states, dones = buffer.sample(32)
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # return as tensors when needed; keep here as numpy arrays
        states = np.stack([t.state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
