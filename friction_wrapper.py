"""
Friction Wrapper for Panda-Gym Pick and Place Environment.

This wrapper randomizes the lateral friction of the target object at each episode reset,
making friction observable to the agent for domain randomization training.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FrictionWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that randomizes object friction on each episode reset.
    
    The friction value is added to the observation space so the agent can
    adapt its policy based on the current friction level.
    
    Args:
        env: The Panda-Gym environment to wrap (e.g., PandaPickAndPlace-v3)
        friction_range: Tuple of (min, max) friction values. Default (0.1, 2.0)
    """
    
    def __init__(self, env: gym.Env, friction_range: tuple = (0.1, 2.0)):
        super().__init__(env)
        self.friction_min, self.friction_max = friction_range
        self.current_friction = 0.5  # Default friction
        
        # Extend observation space to include friction
        # Panda-Gym uses Dict observation space with 'observation', 'achieved_goal', 'desired_goal'
        original_obs_space = env.observation_space
        
        # Create new observation space with friction added
        new_spaces = {}
        for key, space in original_obs_space.spaces.items():
            new_spaces[key] = space
        
        # Add friction as a new observation key
        new_spaces['friction'] = spaces.Box(
            low=np.array([self.friction_min], dtype=np.float32),
            high=np.array([self.friction_max], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(new_spaces)
    
    def reset(self, **kwargs):
        """Reset environment and randomize object friction."""
        obs, info = self.env.reset(**kwargs)
        
        # Sample random friction value
        self.current_friction = np.random.uniform(self.friction_min, self.friction_max)
        
        # Set friction on the target object using panda-gym's built-in method
        # 'object' is the target object, -1 is the base link
        self.env.unwrapped.sim.set_lateral_friction('object', -1, self.current_friction)
        
        # Add friction to observation
        obs['friction'] = np.array([self.current_friction], dtype=np.float32)
        
        # Also add to info for logging purposes
        info['friction'] = self.current_friction
        
        return obs, info
    
    def step(self, action):
        """Step environment, adding friction to observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add current friction to observation
        obs['friction'] = np.array([self.current_friction], dtype=np.float32)
        
        # Add to info for logging
        info['friction'] = self.current_friction
        
        return obs, reward, terminated, truncated, info


class FixedFrictionWrapper(gym.Wrapper):
    """
    A wrapper that sets a fixed friction value (for evaluation at specific friction levels).
    
    Args:
        env: The Panda-Gym environment to wrap
        friction: The fixed friction value to use
    """
    
    def __init__(self, env: gym.Env, friction: float = 0.5):
        super().__init__(env)
        self.friction = friction
        self.current_friction = friction
        
        # Extend observation space to include friction
        original_obs_space = env.observation_space
        new_spaces = {}
        for key, space in original_obs_space.spaces.items():
            new_spaces[key] = space
        
        new_spaces['friction'] = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(new_spaces)
    
    def reset(self, **kwargs):
        """Reset environment and set fixed friction."""
        obs, info = self.env.reset(**kwargs)
        
        # Set fixed friction on the target object
        self.env.unwrapped.sim.set_lateral_friction('object', -1, self.friction)
        
        # Add friction to observation
        obs['friction'] = np.array([self.current_friction], dtype=np.float32)
        info['friction'] = self.current_friction
        
        return obs, info
    
    def step(self, action):
        """Step environment, adding friction to observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs['friction'] = np.array([self.current_friction], dtype=np.float32)
        info['friction'] = self.current_friction
        return obs, reward, terminated, truncated, info

