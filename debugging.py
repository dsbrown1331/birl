import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import numpy as np
from mdp import FeatureMDP

if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95
    num_rows = 4 # 4 normal, 5 driving
    num_cols = 4
    num_features = 3

    # MCMC hyperparameters
    beta = 10.0  # confidence for mcmc
    N = 50
    step_stdev = 0.4
    burn_rate = 0.0
    skip_rate = 1
    random_normalization = True # whether or not to normalize with random policy

    # Experiment setup
    threshold = 0.0
    demo_order = list(range(num_rows * num_cols))
    random.shuffle(demo_order)

    all_accept_rates = 0
    accept_rate_counts = 0

    # env = FeatureMDP(num_rows, num_cols, 4, [], np.array([-0.1699364,  -0.86117737, -0.47905652]), np.array([(0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]), gamma)
    env = mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features)
    opt_policy = mdp_utils.get_optimal_policy(env)
    if debug:
        print("PROBLEM WORLD")
        mdp_utils.visualize_binary_features(env)
    baseline_pi = mdp_utils.get_nonpessimal_policy(env)
    if debug:
        print("Baseline policy")
        mdp_utils.visualize_policy(baseline_pi, env)
        policy_optimality = mdp_utils.calculate_percentage_optimal_actions(baseline_pi, env)
        print("Baseline policy optimality:", policy_optimality)
        policy_accuracy = mdp_utils.calculate_policy_accuracy(opt_policy, baseline_pi)
        print("Baseline policy accuracy:", policy_accuracy)
    demos = []
    for M in range(len(demo_order)):
        D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
        demos.append(D)
        if debug:
            print("Running BIRL with {} demos:".format(M), demos)
        birl = bayesian_irl.BIRL(env, demos, beta) # create BIRL environment
        # use MCMC to generate sequence of sampled rewards
        birl.run_mcmc(N, step_stdev, adaptive=True)
        #burn initial samples and skip every skip_rate for efficiency
        burn_indx = int(len(birl.chain) * burn_rate)
        samples = birl.chain[burn_indx::skip_rate]
        #check if MCMC seems to be mixing properly
        all_accept_rates += birl.accept_rate
        accept_rate_counts += 1
        if debug:
            avg_accept_rate = all_accept_rates / accept_rate_counts
            if avg_accept_rate > 0.7:
                msg = ", too high, probably need to increase stdev"
            elif avg_accept_rate < 0.2:
                msg = ", too low, probably need to decrease stdev"
            else:
                msg = ""
            print("accept rate: " + str(birl.accept_rate) + "; avg accept rate: " + str(avg_accept_rate) + msg) #good to tune number of samples and stepsize to have this around 50%
        #generate evaluation policy from running BIRL
        map_env = copy.deepcopy(env)
        map_env.set_rewards(birl.get_map_solution())
        map_policy = mdp_utils.get_optimal_policy(map_env)
        #debugging to visualize the learned policy
        if debug:
            print("True weights", env.feature_weights)
            print("True policy")
            mdp_utils.visualize_policy(opt_policy, env)
            print("MAP weights", map_env.feature_weights)
            print("MAP policy")
            mdp_utils.visualize_policy(map_policy, map_env)
            policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
            print("Policy optimality:", policy_optimality)
            policy_accuracy = mdp_utils.calculate_policy_accuracy(opt_policy, map_policy)
            print("Policy accuracy:", policy_accuracy)

        # get percent improvement
        V_base, V_eval, improvement = mdp_utils.calculate_percent_improvement(env, baseline_pi, map_policy)
        if debug:
            print(V_base, "==>", V_eval)
            print("Improvement over baseline:", improvement)