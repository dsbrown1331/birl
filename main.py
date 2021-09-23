import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np

if __name__ == "__main__":

    nx = 8 # columns
    ny = 6 # rows
    gamma = 0.95
    noise = 0.1
    print("create a random ny x nx gridworld with no featurized reward function")
    terminal_states = [0]#[rows * columns - 1]
    # rewards negative everywhere except for the goal state
    rewards = - np.ones(ny * nx)
    rewards[0] = 2.0
    # rewards[5] = - 1000
    # rewards[6] = - 1000
    # rewards[7] = - 1000
    # constraints is a binary vector indicating hte presence or not of a constraint
    constraints = np.zeros(ny * nx)
    # set some constrains by hand
    constraints[[16, 17, 24, 25, 23, 31]] = 1
    rewards[[16, 17, 24, 25, 23, 31]] = -10
    num_cnstr = len(np.nonzero(constraints)[0])

    env2 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_states, rewards, constraints, gamma, noise)
    np.save('../birl-v2/original', env2.transitions)
    
    # IPython.embed()
    #visualize rewards
    print("random rewards")
    mdp_utils.print_array_as_grid(env2.rewards, env2)
    print("optimal policy for random grid world")
    mdp_utils.visualize_policy(mdp_utils.get_optimal_policy(env2), env2)
    optimal_policy = mdp_utils.get_optimal_policy(env2)
    # plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, optimal_policy)
    bottom_right_corner = ny * nx - 1
    trajectory_demos = []
    boltz_beta = 1.0


    for i in range(40):
        # trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_left_corner))
        trajectory_demos.append(mdp_utils.generate_boltzman_demo(env2, boltz_beta, bottom_right_corner))
    
    trajectory_demos = [item for sublist in trajectory_demos for item in sublist]
    print("optimal trajectory starting from state 2")
    print(trajectory_demos)
    #format with arrows
    mdp_utils.visualize_trajectory(trajectory_demos, env2)
    
    env_orig = copy.deepcopy(env2)

    print("Running Bayesian IRL with full optimal policy as demo")
    demos = mdp_utils.demonstrate_entire_optimal_policy(env2)
    # IPython.embed()
    print(demos)
    # second argument is the demonstrations
    # 1) demos is for all states, 2) trajectory_demos is just some trajectories
    # birl = bayesian_irl.BIRL(env2, trajectory_demos, beta, num_cnstr=num_cnstr)
    # num_steps = 10000
    # step_stdev = 0.1
    rewards_fix = - np.ones(ny * nx)
    rewards_fix[0] = 2.0
    # birl.run_mcmc_bern(num_steps, 0.5, rewards_fix)
    # map_constr, map_rew = birl.get_map_solution()
    constraints_map = np.zeros(ny * nx)
    # # set some constrains by hand
    # # for i in range(len(map_reward)):
    # #     if map_reward[i] < - 1.1:
    # #         constraints_map[i] = 1

    constraints[[16, 17, 24, 25, 23, 31]] = 1
    # mean_constraints = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
    # # plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, None, mean_constraints, trajectory_demos, optimal_policy)
    # plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, trajectory_demos, optimal_policy)
    # plot_grid.plot_grid(nx, ny, env2.state_grid_map, map_constr, trajectory_demos, optimal_policy)
    
    # plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, map_constr, mean_constraints, trajectory_demos, optimal_policy)
            
    # # print("map reward", map_reward)
    # mean_cnstr = birl.get_mean_solution(burn_frac=0.1, skip_rate=1)
    # print("mean reward", mean_cnstr)
    # IPython.embed()
    for kk in range(20):
        birl = bayesian_irl.BIRL(env2, trajectory_demos, boltz_beta, env_orig)

     
        


        num_steps = 4000
        step_stdev = 0.1
        # for _ in range(5):
        # birl.run_mcmc_bern_reward(num_steps, 0.5, env2.rewards)
        birl.run_mcmc_bern_constraint(num_steps, 0.5, rewards_fix)






        map_constraints = birl.get_map_solution()
        constraints_map = np.zeros(ny * nx)
        # set some constrains by hand
        # for i in map_constraints:
        #         constraints_map[i] = 1
        # constraints_map  = map_constraints
        acc_rate = birl.accept_rate
        # constraints[[16, 17, 24, 25, 23, 31]] = 1
        mean_cnstr = birl.get_mean_solution(burn_frac=0.1, skip_rate=1)
        # plot_grid.plot_posterior(nx, ny, mean_cnstr, kk, constraints, True, env2.state_grid_map)
        # print("map constraints", map_constraints)
        mean_constraints, mean_penalty = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
        # print("mean constraints", mean_constraints)
        map_sol, map_rew = birl.get_map_solution()
        print("Mean results constr", mean_constraints)
        print("Mean results penalty", mean_penalty)
        # print("map constraints probability", map_prob)
        plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, constraints, mean_constraints, trajectory_demos, optimal_policy)
        plot_grid.plot_grid(nx, ny, env2.state_grid_map, kk, constraints, trajectory_demos, optimal_policy)
        # plot_grid.plot_grid_temp2(nx, ny, env2.state_grid_map, constraints_plot, constraints_map, trajectory_demos, optimal_policy, kk)
        np.savetxt('penalty_rew' + str(kk) + '.txt', birl.chain_rew)
        temp_query_state = birl.compute_variance(birl.posterior, 0.1)
        # IPython.embed()
        # if query_state != None:
        #     temp_query_state = list(birl.env.state_grid_map.keys())[list(birl.env.state_grid_map.values()).index(query_state)]
        print('******',temp_query_state,'*****')
        if temp_query_state != None:# and temp_query_state != 0 and temp_query_state not in env_orig.constraints:
            # temp_query_state = list(self.env.state_grid_map.keys())[list(self.env.state_grid_map.values()).index(query_state)]
            
            for _ in range(10):
                traj = mdp_utils.generate_boltzman_demo(env_orig, boltz_beta, temp_query_state)
                query_state_action = traj[0]
                trajectory_demos.append(query_state_action)




# #-----------------------------------    
#     print()
#     print("create simple featurized mdp")
#     env = mdp_worlds.gen_simple_world()
#     #get optimal policy 
#     opt_pi = mdp_utils.get_optimal_policy(env)
#     #text-based visualization of optimal policy
#     print("optimal policy for featurized grid world") 
#     mdp_utils.visualize_policy(opt_pi, env)


# #-----------------------------------
#     print()
#     print("generate a demonstration starting from top right state (states are numbered 0 through num_states")
#     opt_traj = mdp_utils.generate_optimal_demo(env, 2)
#     print("optimal trajectory starting from state 2")
#     print(opt_traj)
#     #format with arrows
#     mdp_utils.visualize_trajectory(opt_traj, env)

# #-----------------------------------
#     print()
#     print("Bayesian IRL")
#     #give an entire policy as a demonstration to Bayesian IRL (best case scenario, jjst to test that BIRL works)
#     print("Running Bayesian IRL with full optimal policy as demo")
#     demos = mdp_utils.demonstrate_entire_optimal_policy(env)
#     print(demos)
#     beta = 10.0 #assume near optimal demonstrator
#     birl = bayesian_irl.BIRL(env, demos, beta)
#     num_steps = 2000
#     step_stdev = 0.1
#     birl.run_mcmc(num_steps, step_stdev)
#     map_reward = birl.get_map_solution()
#     print("map reward", map_reward)
#     mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
#     print("mean reward", mean_reward)

#     # visualize the optimal policy for the learned reward function
#     env_learned = copy.deepcopy(env)
#     env_learned.set_rewards(map_reward)
#     learned_pi = mdp_utils.get_optimal_policy(env_learned)
#     #text-based visualization of optimal policy
#     print("Learned policy from Bayesian IRL using MAP reward") 
#     mdp_utils.visualize_policy(learned_pi, env_learned)
#     print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
#     #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

#     #-----------------------------------
#     print()
#     ## Quantitative metrics
#     print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
#     print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))


#     #-----------------------------------
#     print()
#     ## Bayesian IRL with a demonstration purposely chosen to not be super informative
#     print("running Bayesian IRL with the following demo")
#     opt_traj = mdp_utils.generate_optimal_demo(env, 7)
#     print("optimal trajectory starting from state 7")
#     print(opt_traj)
#     #format with arrows
#     mdp_utils.visualize_trajectory(opt_traj, env)


#     birl = bayesian_irl.BIRL(env, opt_traj, beta)
#     num_steps = 2000
#     step_stdev = 0.1
#     birl.run_mcmc(num_steps, step_stdev)
#     map_reward = birl.get_map_solution()
#     print("map reward", map_reward)
#     mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
#     print("mean reward", mean_reward)

#     # visualize the optimal policy for the learned reward function
#     env_learned = copy.deepcopy(env)
#     env_learned.set_rewards(map_reward)
#     learned_pi = mdp_utils.get_optimal_policy(env_learned)
#     #text-based visualization of optimal policy
#     print("Learned policy from Bayesian IRL using MAP reward") 
#     mdp_utils.visualize_policy(learned_pi, env_learned)
#     print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
#     #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

#     #-----------------------------------
#     print()
#     ## Quantitative metrics
#     print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
#     print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))

#     #demo isn't super informative so it doesn't learn the true optimal policy but does learn that the red feature is probably negative
#     # and likely worse than the white feature

#     print("learned weights", map_reward)
#     print("true weights", env.feature_weights)



