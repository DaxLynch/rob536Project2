"""
Evaluate a trained model's performance across varying friction levels.

This script tests how well a model (trained on standard friction) performs
when the target object has different friction coefficients.

Usage:
    python evaluate_friction.py --model path/to/model.zip
    python evaluate_friction.py --model path/to/model.zip --episodes 50 --render
"""
import argparse
import csv
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import panda_gym
from sbx import TQC

from friction_wrapper import FrictionWrapper, FixedFrictionWrapper


def evaluate_with_random_friction(model, env_name: str, n_episodes: int = 100, 
                                   render: bool = False) -> dict:
    """
    Evaluate model with random friction each episode.
    
    Returns:
        Dictionary with results including success rate, rewards, and friction values
    """
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    env = FrictionWrapper(env)
    
    results = {
        'episodes': [],
        'frictions': [],
        'rewards': [],
        'successes': [],
        'lengths': []
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        friction = info.get('friction', 0.5)
        
        total_reward = 0
        done = False
        length = 0
        
        while not done:
            # Handle observation format for models trained without friction
            if 'friction' in obs and not hasattr(model, '_friction_aware'):
                # Model was trained without friction observation
                # Remove friction from obs for prediction
                obs_for_model = {k: v for k, v in obs.items() if k != 'friction'}
            else:
                obs_for_model = obs
            
            action, _ = model.predict(obs_for_model, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
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


def evaluate_at_fixed_frictions(model, env_name: str, friction_values: list,
                                 n_episodes_per_friction: int = 20,
                                 render: bool = False) -> dict:
    """
    Evaluate model at specific friction values.
    
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
    
    for friction in friction_values:
        if render:
            env = gym.make(env_name, render_mode="human")
        else:
            env = gym.make(env_name)
        env = FixedFrictionWrapper(env, friction=friction)
        
        rewards = []
        successes = []
        lengths = []
        
        for ep in range(n_episodes_per_friction):
            obs, info = env.reset()
            total_reward = 0
            done = False
            length = 0
            
            while not done:
                # Handle observation format for models trained without friction
                if 'friction' in obs:
                    obs_for_model = {k: v for k, v in obs.items() if k != 'friction'}
                else:
                    obs_for_model = obs
                
                action, _ = model.predict(obs_for_model, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
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
        
        print(f"Friction={friction:.2f}: Success Rate={success_rate:.1f}%, "
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
    parser.add_argument('--env', type=str, default='PandaPickAndPlace-v3',
                        help='Environment name (default: PandaPickAndPlace-v3)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'fixed', 'both'],
                        help='Evaluation mode: random (varying each episode), fixed (test specific values), or both')
    parser.add_argument('--output-dir', type=str, default='./friction_eval_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = TQC.load(args.model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Friction Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Environment: {args.env}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    if args.mode in ['random', 'both']:
        print("\n--- Random Friction Evaluation ---\n")
        random_results = evaluate_with_random_friction(
            model, args.env, args.episodes, args.render
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
        friction_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        fixed_results = evaluate_at_fixed_frictions(
            model, args.env, friction_values, 
            n_episodes_per_friction=20, render=args.render
        )
        
        print(f"\n{'='*40}")
        print("Fixed Friction Summary:")
        for i, f in enumerate(fixed_results['friction_levels']):
            print(f"  Friction {f:.1f}: {fixed_results['success_rates'][i]:.1f}% success")
        print(f"{'='*40}")
        
        output_path = os.path.join(args.output_dir, f"fixed_friction_{timestamp}.csv")
        save_results(fixed_results, output_path)


if __name__ == "__main__":
    main()

