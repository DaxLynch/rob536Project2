Friction Evaluation Results
===========================

CSV File Types
--------------
1. *-varying_friction_tests.csv (random friction each episode)
   Columns: episode, friction, reward, success, length
   
2. *-fixed_friction_tests.csv (aggregated stats per friction level)
   Columns: friction, success_rate, mean_reward, std_reward, mean_length


Results Files
-------------

JOINTS CONTROL (8-dim action):
  joints-ConstantFrictionModel-varying_friction_tests.csv
    - Model: constant_friction_joints_mode.zip (trained at friction=0.5)
    - Test: 200 episodes with random friction (0.1-2.0)
    - Result: 51% success rate, struggles with high friction (>1.0)

  joints-ConstantFrictionModel-fixed_friction_tests.csv
    - Same model tested at specific friction values
    - Sweet spot: 0.3-0.7 friction (95-100% success)
    - Degrades at extremes: <0.2 (25-45%) and >0.9 (70-75%)

END-EFFECTOR CONTROL (4-dim action):
  vanillaModelVaryingFriction.csv
    - Model: pick_and_place_end_effector_mode_std_friction.zip
    - 19-dim obs (no friction awareness)

  frictionAwareModelVaryingFriction.csv  
    - Model: pick_and_place_end_effector_friction_mode.zip
    - 20-dim obs (friction-aware)

  *SingleRun.csv files contain per-episode data for above
