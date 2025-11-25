"""
Native Friction Pick and Place Environment for Panda-Gym.

This module provides a custom environment with friction as part of the native
observation, avoiding wrapper overhead that causes slowdowns with JAX-based training.

Environments:
    - FrictionPickAndPlace-v1: Random friction each episode (0.1-2.0)
    - ConstantFrictionPickAndPlace-v1: Fixed friction at 0.5 (for pre-training)
"""
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
#I Originally used a wrapper, and it decrease the performance by like 20x, so I 
#just converted it to a native environment. and this had much better performance.

class PickAndPlaceWithFriction(Task):
    """Pick and Place task with friction as part of the observation.
    
    This task extends the standard PickAndPlace by:
    - Randomizing object friction on each reset
    - Including friction value in the observation array
    
    Args:
        sim: PyBullet simulation instance
        reward_type: "sparse" or "dense"
        friction_range: Tuple of (min, max) friction values
        randomize_friction: If True, randomize friction each reset. If False, use middle of range.
    """
    
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        friction_range: tuple = (0.05, 2.0),
        randomize_friction: bool = True,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        
        # Friction configuration
        self.friction_min, self.friction_max = friction_range
        self.randomize_friction = randomize_friction
        self.current_friction = (self.friction_min + self.friction_max) / 2  # Default to middle
        
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        """Return observation including friction value."""
        # Original observations: position, rotation, velocity, angular velocity of object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        
        # Append friction to observation (11 floats total instead of 10)
        observation = np.concatenate([
            object_position, 
            object_rotation, 
            object_velocity, 
            object_angular_velocity,
            [self.current_friction]  # Friction as the last element
        ])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        """Reset the task and randomize friction."""
        # Randomize friction if enabled
        if self.randomize_friction:
            self.current_friction = self.np_random.uniform(self.friction_min, self.friction_max)
        else:
            self.current_friction = (self.friction_min + self.friction_max) / 2
        
        # Sample goal and object position
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        
        # Set poses
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        
        # Apply friction to object
        self.sim.set_lateral_friction("object", -1, self.current_friction)

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)


class FrictionPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task with friction randomization.
    
    The friction value is included in the observation space and randomized
    each episode. This is a native implementation that avoids wrapper overhead.
    
    Args:
        render_mode: Render mode ("rgb_array" or "human")
        reward_type: "sparse" or "dense"
        control_type: "ee" (end-effector) or "joints"
        friction_range: Tuple of (min, max) friction values
        randomize_friction: If True, randomize friction each reset
        renderer: "Tiny" or "OpenGL"
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        friction_range: tuple = (0.1, 2.0),
        randomize_friction: bool = True,
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceWithFriction(
            sim, 
            reward_type=reward_type,
            friction_range=friction_range,
            randomize_friction=randomize_friction,
        )
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
    
    def get_friction(self) -> float:
        """Return the current friction value."""
        return self.task.current_friction


class ConstantFrictionPickAndPlaceEnv(FrictionPickAndPlaceEnv):
    """Pick and Place with constant friction (for pre-training).
    
    Uses friction=0.5 (the panda-gym default) but includes friction in the
    observation space. Useful for pre-training a model that can later be
    fine-tuned with randomized friction.
    """
    
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        friction: float = 0.5,
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        # Use a tiny range centered on the fixed friction value
        super().__init__(
            render_mode=render_mode,
            reward_type=reward_type,
            control_type=control_type,
            friction_range=(friction, friction),  # Fixed friction
            randomize_friction=False,
            renderer=renderer,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


# Register environments with gymnasium
gym.register(
    id="FrictionPickAndPlace-v1",
    entry_point="friction_env:FrictionPickAndPlaceEnv",
    max_episode_steps=50,
)

gym.register(
    id="ConstantFrictionPickAndPlace-v1",
    entry_point="friction_env:ConstantFrictionPickAndPlaceEnv",
    max_episode_steps=50,
)



