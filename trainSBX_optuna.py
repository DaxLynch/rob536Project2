"""
Optuna Hyperparameter Optimization for TQC on PandaPickAndPlace-v3.

This script uses Optuna to find optimal hyperparameters for TQC training,
following the RL Baselines3 Zoo pattern.

Usage:
    python trainSBX_optuna.py

After optimization completes, use the best hyperparameters in trainSBX.py
"""
import os
import pickle as pkl
import time
from datetime import datetime
from pprint import pprint
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import optuna
import panda_gym
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from sbx import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from friction_wrapper import FrictionWrapper

# ============================================================================
# Optuna Configuration
# ============================================================================
ENV_NAME = "PandaPickAndPlace-v3"
USE_FRICTION = False  # Set to True to optimize with friction randomization
ALGO_NAME = "tqc_sbx_optuna"

# Optuna settings
N_TRIALS = 50  # Number of optimization trials
N_STARTUP_TRIALS = 5  # Trials before pruning starts
N_EVALUATIONS = 4  # Intermediate evaluations per trial (for pruning)
N_EVAL_EPISODES = 5  # Episodes per evaluation

# Training settings per trial (shorter than full training for faster optimization)
OPTUNA_TIMESTEPS = 100_000  # Timesteps per trial
N_ENVS = 8  # Fewer parallel envs during optimization for faster iteration

# Storage for distributed optimization (optional)
STORAGE = None  # e.g., "sqlite:///optuna_study.db" or "optuna_tqc.log"
STUDY_NAME = f"tqc_{ENV_NAME}"

# Create log directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/{ALGO_NAME}/{ENV_NAME}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)


# ============================================================================
# TrialEvalCallback - Reports to Optuna and enables pruning
# ============================================================================
class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial to Optuna.
    Extends EvalCallback to report intermediate rewards and support pruning.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # Report current mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Check if trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# ============================================================================
# Hyperparameter Sampler for TQC
# ============================================================================
def sample_tqc_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sample hyperparameters for TQC using Optuna.
    
    Based on RL Zoo3's sample_tqc_params/sample_sac_params.
    """
    # Sample raw values
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    batch_size_pow = trial.suggest_int("batch_size_pow", 5, 11)  # 32 to 2048
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)
    
    # Network architecture
    net_arch_type = trial.suggest_categorical(
        "net_arch", ["small", "medium", "big", "large", "verybig"]
    )
    
    # TQC-specific parameters
    n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
    top_quantiles_to_drop = trial.suggest_int(
        "top_quantiles_to_drop_per_net", 0, min(n_quantiles - 1, 5)
    )
    
    # Convert to actual values
    gamma = 1 - one_minus_gamma
    batch_size = 2 ** batch_size_pow
    
    # Network architecture mapping
    net_arch_map = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "large": [256, 256, 256],
        "verybig": [512, 512, 512],
    }
    net_arch = net_arch_map[net_arch_type]
    
    # Store human-readable values as user attributes
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("net_arch_list", net_arch)
    
    # Build hyperparameters dict
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "train_freq": train_freq,
        "gradient_steps": train_freq,  # Match gradient steps to train_freq
        "policy_kwargs": {
            "net_arch": net_arch,
            "n_quantiles": n_quantiles,
        },
        "top_quantiles_to_drop_per_net": top_quantiles_to_drop,
    }
    
    return hyperparams


# ============================================================================
# Objective Function for Optuna
# ============================================================================
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    
    Creates a TQC model with sampled hyperparameters, trains it,
    and returns the mean reward for optimization.
    """
    # Sample hyperparameters
    sampled_hyperparams = sample_tqc_params(trial)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters:")
    pprint(sampled_hyperparams)
    print(f"{'='*60}\n")
    
    # Create training environment (no logging to avoid file conflicts)
    env = make_vec_env(
        ENV_NAME,
        n_envs=N_ENVS,
        wrapper_class=FrictionWrapper if USE_FRICTION else None,
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        ENV_NAME,
        n_envs=1,
        wrapper_class=FrictionWrapper if USE_FRICTION else None,
    )
    
    # Calculate evaluation frequency
    eval_freq = max(OPTUNA_TIMESTEPS // N_EVALUATIONS // N_ENVS, 1)
    
    # Create trial evaluation callback
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq,
        deterministic=True,
        verbose=0,
    )
    
    # Create TQC model with sampled hyperparameters
    try:
        model = TQC(
            policy="MultiInputPolicy",
            env=env,
            buffer_size=100_000,  # Smaller buffer for optimization
            verbose=0,
            tensorboard_log=None,  # Disable tensorboard during optimization
            device="cuda",
            **sampled_hyperparams,
        )
        
        # Train the model
        model.learn(
            total_timesteps=OPTUNA_TIMESTEPS,
            callback=eval_callback,
            progress_bar=False,
        )
        
    except (AssertionError, ValueError) as e:
        # Some hyperparameter combinations may cause errors (e.g., NaN)
        print(f"Trial {trial.number} failed with error: {e}")
        env.close()
        eval_env.close()
        raise optuna.exceptions.TrialPruned()
    
    # Clean up
    env.close()
    eval_env.close()
    
    # Check if pruned
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    # Return the last mean reward
    reward = eval_callback.last_mean_reward
    
    print(f"\nTrial {trial.number} finished with reward: {reward:.2f}\n")
    
    del model
    
    return reward


# ============================================================================
# Main Optimization Loop
# ============================================================================
def run_optimization():
    """Run the Optuna hyperparameter optimization."""
    
    print("=" * 80)
    print("Optuna Hyperparameter Optimization for TQC")
    print("=" * 80)
    print(f"  Environment: {ENV_NAME}")
    print(f"  Friction Randomization: {USE_FRICTION}")
    print(f"  Number of Trials: {N_TRIALS}")
    print(f"  Startup Trials (before pruning): {N_STARTUP_TRIALS}")
    print(f"  Evaluations per Trial: {N_EVALUATIONS}")
    print(f"  Timesteps per Trial: {OPTUNA_TIMESTEPS:,}")
    print(f"  Parallel Environments: {N_ENVS}")
    print(f"  Log Directory: {log_dir}")
    print("=" * 80)
    
    # Create sampler and pruner
    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=42,
        multivariate=True,
    )
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_EVALUATIONS // 3,
    )
    
    # Handle storage (for distributed/persistent optimization)
    storage = STORAGE
    if storage is not None and storage.endswith(".log"):
        os.makedirs(os.path.dirname(storage) or ".", exist_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage),
        )
    
    # Create or load study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
        direction="maximize",  # Maximize reward
    )
    
    print(f"\nSampler: TPE - Pruner: Median")
    print(f"Starting optimization...\n")
    
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    print("\n--- Best Trial ---")
    best_trial = study.best_trial
    print(f"Value (Mean Reward): {best_trial.value:.2f}")
    
    print("\nBest Hyperparameters (raw Optuna values):")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    print("\nBest Hyperparameters (converted):")
    for key, value in best_trial.user_attrs.items():
        print(f"    {key}: {value}")
    
    # Generate code snippet for trainSBX.py
    print("\n" + "=" * 80)
    print("Copy this to trainSBX.py:")
    print("=" * 80)
    
    gamma = best_trial.user_attrs.get("gamma", 1 - best_trial.params["one_minus_gamma"])
    batch_size = best_trial.user_attrs.get("batch_size", 2 ** best_trial.params["batch_size_pow"])
    net_arch = best_trial.user_attrs.get("net_arch_list", [256, 256])
    
    print(f"""
model = TQC(
    policy="MultiInputPolicy",
    env=env,
    learning_rate={best_trial.params['learning_rate']:.6f},
    buffer_size=1_000_000,
    batch_size={batch_size},
    gamma={gamma:.6f},
    tau={best_trial.params['tau']:.6f},
    train_freq={best_trial.params['train_freq']},
    gradient_steps={best_trial.params['train_freq']},
    top_quantiles_to_drop_per_net={best_trial.params['top_quantiles_to_drop_per_net']},
    policy_kwargs=dict(
        net_arch={net_arch},
        n_quantiles={best_trial.params['n_quantiles']},
    ),
    verbose=1,
    tensorboard_log=tensorboard_log,
    device="cuda",
)
""")
    
    # Save reports
    report_name = f"report_{ENV_NAME}_{N_TRIALS}trials_{int(time.time())}"
    report_path = os.path.join(log_dir, report_name)
    
    # Save CSV report
    study.trials_dataframe().to_csv(f"{report_path}.csv")
    print(f"\nCSV report saved to: {report_path}.csv")
    
    # Save study object for later analysis
    with open(f"{report_path}.pkl", "wb") as f:
        pkl.dump(study, f)
    print(f"Study object saved to: {report_path}.pkl")
    
    # Try to show plots (requires plotly)
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html(f"{report_path}_history.html")
        print(f"Optimization history plot saved to: {report_path}_history.html")
        
        fig2 = plot_param_importances(study)
        fig2.write_html(f"{report_path}_importances.html")
        print(f"Parameter importances plot saved to: {report_path}_importances.html")
    except (ValueError, ImportError, RuntimeError) as e:
        print(f"\nCould not generate plots: {e}")
    
    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)
    
    return study


if __name__ == "__main__":
    run_optimization()

