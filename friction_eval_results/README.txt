Friction Evaluation Results
===========================

Two types of CSV files:

1. random_friction_*.csv
   - One row per episode
   - Columns: episode, friction, reward, success, length
   - Friction is randomized (0.1-2.0) each episode

2. fixed_friction_*.csv
   - One row per friction level tested
   - Columns: friction, success_rate, mean_reward, std_reward, mean_length
   - Aggregated stats from 20 episodes per friction value



