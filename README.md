# Demonstration Sufficiency

`sufficiency.py` holds the main experiments for testing the stopping conditions for demonstration sufficiency.

How to run:
```python <stopping_condition> <environment> <demonstration_type>```

where `stopping_condition` can be `nevd` for the nEVD stopping condition or `baseline_pi` for the percent improvement over a baseline stopping condition, `environment` can be `goal` for the gridworld or `driving` for the driving, and `demonstration_type` can be `pairs` or `trajectories` (we used pairs).
