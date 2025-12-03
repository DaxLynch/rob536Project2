"""
Evaluate a trained model's performance across varying friction levels.

This script tests how well a model performs when the target object has 
different friction coefficients.

Supports:
- End-effector control (4-dim action) or joints control (7-dim action)
- Friction-aware (20-dim obs) or standard (19-dim obs)

Usage:
    python evaluate_friction.py --model path/to/model.zip
    python evaluate_friction.py --model path/to/model.zip --control joints
    python evaluate_friction.py --model path/to/model.zip --episodes 50 --render
"""
import argparse
import csv
import os
from datetime import datetime
import time
import gymnasium as gym
import numpy as np
import panda_gym
from sbx import TQC

# Import to register native friction environments
import friction_env


def evaluate_with_random_friction(model, n_episodes: int = 100, 
                                   render: bool = False,
                                   use_friction_env: bool = True,
                                   control_type: str = "ee") -> dict:
    """
    Evaluate model with random friction each episode.
    
    Args:
        model: Trained model
        n_episodes: Number of episodes to run
        render: Whether to render
        use_friction_env: If True, use friction env (friction-aware models)
                         If False, use standard env and manually set friction
        control_type: "ee" for end-effector, "joints" for joint control
    
    Returns:
        Dictionary with results including success rate, rewards, and friction values
    """
    # Select environment based on control type and friction awareness
    if control_type == "joints":
        env_name = "FrictionPickAndPlaceJoints-v1" if use_friction_env else "PandaPickAndPlaceJoints-v3"
    else:
        env_name = "FrictionPickAndPlace-v1" if use_friction_env else "PandaPickAndPlace-v3"
    
    if render:
        env = gym.make(env_name, render_mode="human", renderer="OpenGL")
    else:
        env = gym.make(env_name)
    
    results = {
        'episodes': [],
        'frictions': [],
        'rewards': [],
        'successes': [],
        'lengths': []
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        # Set/get friction depending on env type
        if use_friction_env:
            friction = obs['observation'][-1]
        else:
            # Manually set random friction for standard env
            friction = np.random.uniform(0.1, 2.0)
            env.unwrapped.sim.set_lateral_friction('object', -1, friction)
        
        total_reward = 0
        done = False
        length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render:
                time.sleep(1.0/12.0)
            length += 1
            done = terminated or truncated
        
        success = info.get('is_success', False)
        
        results['episodes'].append(ep)
        results['frictions'].append(friction)
        results['rewards'].append(total_reward)
        results['successes'].append(success)
        results['lengths'].append(length)
        
        print(f"Episode {ep+1}/{n_episodes}: Friction={friction:.3f}, "
              f"Reward={total_reward:.2f}, Success={success}, Length={length}")
    
    env.close()
    return results


def evaluate_at_fixed_frictions(model, friction_values: list,
                                 n_episodes_per_friction: int = 20,
                                 render: bool = False,
                                 use_friction_env: bool = True,
                                 control_type: str = "ee") -> dict:
    """
    Evaluate model at specific friction values.
    
    Args:
        model: Trained model
        friction_values: List of friction values to test
        n_episodes_per_friction: Episodes per friction level
        render: Whether to render
        use_friction_env: If True, use friction env (friction-aware models)
                         If False, use standard env and manually set friction
        control_type: "ee" for end-effector, "joints" for joint control
    
    Returns:
        Dictionary with results grouped by friction level
    """
    
    results = {
        'friction_levels': [],
        'success_rates': [],
        'mean_rewards': [],
        'std_rewards': [],
        'mean_lengths': []
    }
    
    # Select environment based on control type
    if control_type == "joints":
        friction_env_name = "FrictionPickAndPlaceJoints-v1"
        standard_env_name = "PandaPickAndPlaceJoints-v3"
    else:
        friction_env_name = "FrictionPickAndPlace-v1"
        standard_env_name = "PandaPickAndPlace-v3"
    
    for friction in friction_values:
        if use_friction_env:
            # Use native friction env with fixed friction
            if render:
                env = gym.make(friction_env_name, 
                              render_mode="human", 
                              renderer="OpenGL",
                              friction_range=(friction, friction),
                              randomize_friction=False)
            else:
                env = gym.make(friction_env_name,
                              friction_range=(friction, friction),
                              randomize_friction=False)
        else:
            # Use standard env, will manually set friction
            if render:
                env = gym.make(standard_env_name, render_mode="human", renderer="OpenGL")
            else:
                env = gym.make(standard_env_name)
        
        rewards = []
        successes = []
        lengths = []
        
        for ep in range(n_episodes_per_friction):
            obs, info = env.reset()
            
            # Manually set friction for standard env
            if not use_friction_env:
                env.unwrapped.sim.set_lateral_friction('object', -1, friction)
            
            total_reward = 0
            done = False
            length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render:
                    time.sleep(1.0/12.0)
                length += 1
                done = terminated or truncated
            
            rewards.append(total_reward)
            successes.append(info.get('is_success', False))
            lengths.append(length)
        
        env.close()
        
        success_rate = np.mean(successes) * 100
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(lengths)
        
        results['friction_levels'].append(friction)
        results['success_rates'].append(success_rate)
        results['mean_rewards'].append(mean_reward)
        results['std_rewards'].append(std_reward)
        results['mean_lengths'].append(mean_length)
        
        print(f"Friction={friction:.3f}: Success Rate={success_rate:.1f}%, "
              f"Mean Reward={mean_reward:.2f} ± {std_reward:.2f}")
    
    return results


def save_results(results: dict, output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if 'episodes' in results:
            # Random friction results
            writer.writerow(['episode', 'friction', 'reward', 'success', 'length'])
            for i in range(len(results['episodes'])):
                writer.writerow([
                    results['episodes'][i],
                    results['frictions'][i],
                    results['rewards'][i],
                    results['successes'][i],
                    results['lengths'][i]
                ])
        else:
            # Fixed friction results
            writer.writerow(['friction', 'success_rate', 'mean_reward', 'std_reward', 'mean_length'])
            for i in range(len(results['friction_levels'])):
                writer.writerow([
                    results['friction_levels'][i],
                    results['success_rates'][i],
                    results['mean_rewards'][i],
                    results['std_rewards'][i],
                    results['mean_lengths'][i]
                ])
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with varying friction')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'fixed', 'both'],
                        help='Evaluation mode: random (varying each episode), fixed (test specific values), or both')
    parser.add_argument('--control', type=str, default='ee', choices=['ee', 'joints'],
                        help='Control type: ee (end-effector, 4-dim action) or joints (7-dim action)')
    parser.add_argument('--output-dir', type=str, default='./friction_eval_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Detect model type by trying to load with each env type
    print(f"Loading model from: {args.model}")
    print(f"  Control type: {args.control}")
    
    # Select environments based on control type
    if args.control == "joints":
        friction_env_name = "FrictionPickAndPlaceJoints-v1"
        standard_env_name = "PandaPickAndPlaceJoints-v3"
    else:
        friction_env_name = "FrictionPickAndPlace-v1"
        standard_env_name = "PandaPickAndPlace-v3"
    
    # First try friction env (20-dim obs)
    try:
        temp_env = gym.make(friction_env_name)
        model = TQC.load(args.model, env=temp_env)
        temp_env.close()
        use_friction_env = True
        env_name = friction_env_name
        print(f"  Model type: Friction-aware (20-dim observation)")
    except ValueError:
        # Fall back to standard env (19-dim obs)
        temp_env = gym.make(standard_env_name)
        model = TQC.load(args.model, env=temp_env)
        temp_env.close()
        use_friction_env = False
        env_name = standard_env_name
        print(f"  Model type: Standard (19-dim observation, friction NOT in obs)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Friction Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Environment: {env_name}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    if args.mode in ['random', 'both']:
        print("\n--- Random Friction Evaluation ---\n")
        random_results = evaluate_with_random_friction(
            model, args.episodes, args.render, use_friction_env, args.control
        )
        
        # Summary
        print(f"\n{'='*40}")
        print("Random Friction Summary:")
        print(f"  Total Episodes: {args.episodes}")
        print(f"  Success Rate: {np.mean(random_results['successes'])*100:.1f}%")
        print(f"  Mean Reward: {np.mean(random_results['rewards']):.2f}")
        print(f"  Friction Range: [{min(random_results['frictions']):.3f}, {max(random_results['frictions']):.3f}]")
        print(f"{'='*40}")
        
        output_path = os.path.join(args.output_dir, f"random_friction_{timestamp}.csv")
        save_results(random_results, output_path)
    
    if args.mode in ['fixed', 'both']:
        print("\n--- Fixed Friction Evaluation ---\n")
        # Test at specific friction values
        friction_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]#1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        fixed_results = evaluate_at_fixed_frictions(
            model, friction_values, 
            n_episodes_per_friction=20, render=args.render,
            use_friction_env=use_friction_env, control_type=args.control
        )
        
        print(f"\n{'='*40}")
        print("Fixed Friction Summary:")
        for i, f in enumerate(fixed_results['friction_levels']):
            print(f"  Friction {f:.3f}: {fixed_results['success_rates'][i]:.1f}% success")
        print(f"{'='*40}")
        
        output_path = os.path.join(args.output_dir, f"fixed_friction_{timestamp}.csv")
        save_results(fixed_results, output_path)


if __name__ == "__main__":
    main()
