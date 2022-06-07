import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math
import sys

rseed = 168
random.seed(rseed)
np.random.seed(rseed)

if __name__ == "__main__":
    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    num_rows = 5 # 4 normal, 5 driving
    num_cols = 4
    num_features = 3
    thresholds = [round(th, 1) for th in np.arange(start = 2.0, stop = -0.1, step = -0.2)] # thresholds on the a-VaR bounds

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    N = 450
    step_stdev = 0.3
    burn_rate = 0.05
    skip_rate = 2
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 20

    # Experiment setup
    envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
    policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
    policy_archive = [{} for _ in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    demo_order = list(range(num_rows * num_cols))
    random.shuffle(demo_order)

    # Metrics to evaluate thresholds
    accuracies = {threshold: 0 for threshold in thresholds}
    avg_bound_errors = {threshold: [] for threshold in thresholds}
    bounds = {threshold: [] for threshold in thresholds}
    true_evds = {threshold: [] for threshold in thresholds}
    num_demos = {threshold: 0 for threshold in thresholds}
    start_comp = 0
    
    for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
        curr_good_upper_bound = 0
        curr_bound_error = []
        curr_p_loss_bound = []
        curr_norm_evd = []
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
                print("map policy")
                print("MAP weights", map_env.feature_weights)
                # mdp_utils.visualize_policy(map_policy, env)
                print("optimal policy")
                print("true weights", env.feature_weights)
                opt_policy = mdp_utils.get_optimal_policy(env)
                # mdp_utils.visualize_policy(opt_policy, env)
                policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                print("policy accuracy", policy_accuracy)

            #run counterfactual policy loss calculations using eval policy
            policy_losses = []
            for sample in samples:
                learned_env = copy.deepcopy(env)
                learned_env.set_rewards(sample)
                # learned_policy = mdp_utils.get_optimal_policy(learned_env)
                Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization) # compute policy loss
                policy_losses.append(Zi)

            #calculate the ground-truth EVD for evaluation
            map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
            curr_norm_evd.append(map_evd)
            policy_losses.sort()
            N_burned = len(samples)
            #compute VaR bound
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
            if k >= len(policy_losses):
                k = len(policy_losses) - 1
            avar_bound = policy_losses[k]
            if debug:
                print("var bound", avar_bound)
            curr_p_loss_bound.append(avar_bound)
            # check if bound is actually upper bound  
            curr_good_upper_bound += avar_bound >= map_evd
            # bound error: upper bound - EVD(eval_policy with ground truth reward)
            curr_bound_error.append(avar_bound - map_evd)
            #debug stuff to see if bounds are good or not
            if debug:
                print("true evd", map_evd)
                print("good", avar_bound >= map_evd)
                print("bound error", avar_bound - map_evd)
        for t in range(len(thresholds[start_comp:])):
            threshold = thresholds[t + start_comp]
            if avar_bound < threshold:
                accuracy = curr_good_upper_bound / num_worlds
                accuracies[threshold] = accuracy
                avg_bound_errors[threshold].append(curr_bound_error)
                bounds[threshold].append(curr_p_loss_bound)
                true_evds[threshold].append(curr_norm_evd)
                num_demos[threshold] = M + 1
            else:
                start_comp += t
                break
    
    # Output results for plotting
    for threshold in thresholds:
        print("NEW THRESHOLD ", threshold)
        print("Accuracy")
        print(accuracies[threshold])
        print("Bound errors")
        for abe in avg_bound_errors[threshold]:
            print(abe)
        print("Policy loss bounds")
        for apl in bounds[threshold]:
            print(apl)
        print("True EVDs")
        for tevd in true_evds[threshold]:
            print(tevd)
        print("Num demos")
        print(num_demos[threshold])
    print("**************************************************")
