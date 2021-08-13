import mdp_utils
import mdp_worlds
import bayesian_irl
import copy


if __name__ == "__main__":


    print("create a random 3x3 gridworld with no featurized reward function")
    env2 = mdp_worlds.random_gridworld(3, 3)
    #visualize rewards
    print("random rewards")
    mdp_utils.print_array_as_grid(env2.rewards, env2)
    print("optimal policy for random grid world")
    mdp_utils.visualize_policy(mdp_utils.get_optimal_policy(env2), env2)

#-----------------------------------    
    print()
    print("create simple featurized mdp")
    env = mdp_worlds.gen_simple_world()
    #get optimal policy 
    opt_pi = mdp_utils.get_optimal_policy(env)
    #text-based visualization of optimal policy
    print("optimal policy for featurized grid world") 
    mdp_utils.visualize_policy(opt_pi, env)


#-----------------------------------
    print()
    print("generate a demonstration starting from top right state (states are numbered 0 through num_states")
    opt_traj = mdp_utils.generate_optimal_demo(env, 2)
    print("optimal trajectory starting from state 2")
    print(opt_traj)
    #format with arrows
    mdp_utils.visualize_trajectory(opt_traj, env)

#-----------------------------------
    print()
    print("Bayesian IRL")
    #give an entire policy as a demonstration to Bayesian IRL (best case scenario, jjst to test that BIRL works)
    print("Running Bayesian IRL with full optimal policy as demo")
    demos = mdp_utils.demonstrate_entire_optimal_policy(env)
    print(demos)
    beta = 10.0 #assume near optimal demonstrator
    birl = bayesian_irl.BIRL(env, demos, beta)
    num_steps = 2000
    step_stdev = 0.1
    birl.run_mcmc(num_steps, step_stdev)
    map_reward = birl.get_map_solution()
    print("map reward", map_reward)
    mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
    print("mean reward", mean_reward)

    # visualize the optimal policy for the learned reward function
    env_learned = copy.deepcopy(env)
    env_learned.set_rewards(map_reward)
    learned_pi = mdp_utils.get_optimal_policy(env_learned)
    #text-based visualization of optimal policy
    print("Learned policy from Bayesian IRL using MAP reward") 
    mdp_utils.visualize_policy(learned_pi, env_learned)
    print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
    #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

    #-----------------------------------
    print()
    ## Quantitative metrics
    print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
    print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))


    #-----------------------------------
    print()
    ## Bayesian IRL with a demonstration purposely chosen to not be super informative
    print("running Bayesian IRL with the following demo")
    opt_traj = mdp_utils.generate_optimal_demo(env, 7)
    print("optimal trajectory starting from state 7")
    print(opt_traj)
    #format with arrows
    mdp_utils.visualize_trajectory(opt_traj, env)


    birl = bayesian_irl.BIRL(env, opt_traj, beta)
    num_steps = 2000
    step_stdev = 0.1
    birl.run_mcmc(num_steps, step_stdev)
    map_reward = birl.get_map_solution()
    print("map reward", map_reward)
    mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
    print("mean reward", mean_reward)

    # visualize the optimal policy for the learned reward function
    env_learned = copy.deepcopy(env)
    env_learned.set_rewards(map_reward)
    learned_pi = mdp_utils.get_optimal_policy(env_learned)
    #text-based visualization of optimal policy
    print("Learned policy from Bayesian IRL using MAP reward") 
    mdp_utils.visualize_policy(learned_pi, env_learned)
    print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
    #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

    #-----------------------------------
    print()
    ## Quantitative metrics
    print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
    print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))

    #demo isn't super informative so it doesn't learn the true optimal policy but does learn that the red feature is probably negative
    # and likely worse than the white feature

    print("learned weights", map_reward)
    print("true weights", env.feature_weights)



