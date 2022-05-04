import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math
import sys

if __name__ == "__main__":
    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    max_num_demos = 9 # maximum number of demos to give agent, start with 1 demo and then work up to max_num_demos
    alpha = float(sys.argv[1])
    delta = 0.05
    num_rows = 4
    num_cols = 4
    num_features = 3

    # MCMC hyperparameters
    beta = 10.0  # confidence for mcmc
    N = 500
    step_stdev = 0.3
    burn_rate = 0.1
    skip_rate = 2
    random_normalization = bool(sys.argv[2]) # whether or not to normalize with random policy
    num_worlds = 100

    envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
    policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
    policy_archive = [{} for _ in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    demo_order = list(range(num_rows * num_cols))
    random.shuffle(demo_order)
    accuracies = []
    avg_bound_errors = []
    bounds = []
    evds = []
    for M in range(0, max_num_demos): # number of demonstrations
        good_upper_bound = 0
        bound_error = []
        p_loss_bound = []
        norm_evd = []
        for i in range(num_worlds):
            env = envs[i]
            archive = policy_archive[i]
            D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
            demos[i].append(D)
            if debug:
                print("running BIRL with demos")
                print("demos", demos[i])
            birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
            # use MCMC to generate sequence of sampled rewards
            birl.run_mcmc(N, step_stdev)
            #burn initial samples and skip every skip_rate for efficiency
            burn_indx = int(len(birl.chain) * burn_rate)
            samples = birl.chain[burn_indx::skip_rate]
            #check if MCMC seems to be mixing properly
            if debug:
                print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
                if birl.accept_rate > 0.7:
                    print("too high, probably need to increase standard deviation")
                elif birl.accept_rate < 0.2:
                    print("too low, probably need to decrease standard dev")
            #generate evaluation policy from running BIRL
            map_env = copy.deepcopy(env)
            map_env.set_rewards(birl.get_map_solution())
            map_policy = mdp_utils.get_optimal_policy(map_env)
            #debugging to visualize the learned policy
            if debug:
                print("feature map")
                mdp_utils.visualize_binary_features(env)
                print("map policy")
                print("MAP weights", map_env.feature_weights)
                mdp_utils.visualize_policy(map_policy, env)
                print("optimal policy")
                print("true weights", env.feature_weights)
                opt_policy = mdp_utils.get_optimal_policy(env)
                mdp_utils.visualize_policy(opt_policy, env)
                policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                print("policy accuracy", policy_accuracy)

            #run counterfactual policy loss calculations using eval policy
            policy_losses = []
            for sample in samples:
                learned_env = copy.deepcopy(env)
                learned_env.set_rewards(sample)
                # learned_policy = mdp_utils.get_optimal_policy(learned_env)
                Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, rn=random_normalization) # compute policy loss
                policy_losses.append(Zi)

            #compute VaR bound
            policy_losses.sort()
            N_burned = len(samples)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
            if k >= len(policy_losses):
                k = len(policy_losses) - 1
            avar_bound = policy_losses[k]
            if debug:
                print("vav bound", avar_bound)
            p_loss_bound.append(avar_bound)
            # accuracy: # of trials where upper bound > ground truth expected value difference / total # of trials
            
            #calculate the ground-truth EVD for evaluation
            map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, rn=random_normalization)
            #check if bound is actually upper bound  
            good_upper_bound += avar_bound >= map_evd
            # bound error: upper bound - EVD(eval_policy with ground truth reward)
            bound_error.append(avar_bound - map_evd)
            #record ground-truth loss based on ground-truth reward function
            norm_evd.append(map_evd)
            #debug stuff to see if bounds are good or not
            if debug:
                print("true evd", map_evd)
                print("good", avar_bound >= map_evd)
                print("bound error", avar_bound - map_evd)
        accuracy = good_upper_bound / num_worlds
        accuracies.append(accuracy)
        avg_bound_errors.append(bound_error)
        bounds.append(p_loss_bound)
        evds.append(norm_evd)
        print("Done with {} demonstrations".format(M + 1))
    print("Random Norm: " + str(random_normalization) + ", Alpha: " + str(alpha))
    print("Accuracies:", accuracies)
    print("Bound errors:", avg_bound_errors)
    print("Policy loss bounds:", bounds)
    print("True EVDs:", evds)
    print("**************************************************")
