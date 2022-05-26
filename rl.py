from mdp import DrivingSimulator
import mdp_utils
import numpy as np
import bayesian_irl
import copy

if __name__ == "__main__":
    print("Creating a driving simulation with NASTY reward function")
    motorists = [1, 7, 11, 13, 17, 23]
    reversed_motorists = [3, 7, 11, 13, 17, 21]
    weights = np.array([1, 1, 1, 10, 5])
    env = DrivingSimulator(5, [], weights / np.linalg.norm(weights), motorists, None, 0.95)
    mdp_utils.visualize_env(env, reversed_motorists)
    #get optimal policy
    nasty_pi = mdp_utils.get_optimal_policy(env)
    #text-based visualization of optimal policy
    print("Learned policy")
    mdp_utils.visualize_policy(mdp_utils.reverse_states(env, nasty_pi), env)
    print()

#-----------------------------------
    print("Creating a driving simulation with ON-ROAD PREFERENCE reward function")
    weights = np.array([10, 10, 10, 0, -20])
    env = DrivingSimulator(5, [], weights / np.linalg.norm(weights), motorists, None, 0.95)
    mdp_utils.visualize_env(env, reversed_motorists)
    #get optimal policy
    pref_pi = mdp_utils.get_optimal_policy(env)
    #text-based visualization of optimal policy
    print("Learned policy")
    mdp_utils.visualize_policy(mdp_utils.reverse_states(env, pref_pi), env)
    print()

#-----------------------------------
    print("Creating a driving simulation with SAFE reward function")
    weights = np.array([5, 5, 10, -20, -10])
    env = DrivingSimulator(5, [], weights / np.linalg.norm(weights), motorists, None, 0.95)
    mdp_utils.visualize_env(env, reversed_motorists)
    #get optimal policy
    safe_pi = mdp_utils.get_optimal_policy(env)
    #text-based visualization of optimal policy
    print("Learned policy")
    mdp_utils.visualize_policy(mdp_utils.reverse_states(env, safe_pi), env)
    print()

#-----------------------------------
    print("Now using only SAFE reward function")
    print("Generating a demonstration starting from middle bottom state")
    opt_traj = mdp_utils.generate_optimal_demo(env, 2)
    print(opt_traj)
    #format with arrows
    print()

#-----------------------------------
    print("Bayesian IRL")
    #give an entire policy as a demonstration to Bayesian IRL (best case scenario, jjst to test that BIRL works)
    print("Running Bayesian IRL with demo using full optimal policy")
    demos = mdp_utils.demonstrate_entire_optimal_policy(env)
    print(demos)
    beta = 10.0 #assume near optimal demonstrator
    birl = bayesian_irl.BIRL(env, demos, beta)
    num_steps = 2000
    step_stdev = 0.1
    birl.run_mcmc(num_steps, step_stdev)
    map_reward = birl.get_map_solution()

    # visualize the optimal policy for the learned reward function
    env_learned = copy.deepcopy(env)
    env_learned.set_rewards(map_reward)
    learned_pi = mdp_utils.get_optimal_policy(env_learned)
    #text-based visualization of optimal policy
    print("Using MAP reward function, BIRL learned this policy") 
    mdp_utils.visualize_policy(learned_pi, env_learned)
    print("MAP reward:", map_reward)
    print("True weights:", env.feature_weights)