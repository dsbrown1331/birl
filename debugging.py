import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import numpy as np
import math
from scipy.stats import norm

if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = "avar"
    world = "feature"

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95
    num_rows = 5 # 4 normal, 5 driving
    num_cols = 5 # 4 normal, 5 driving
    num_features = 4

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    N = 5 # gets around 500 after burn and skip
    step_stdev = 0.5
    burn_rate = 0.00
    skip_rate = 1
    random_normalization = True # whether or not to normalize with random policy
    adaptive = True # whether or not to use adaptive step size
    num_worlds = 5


    # Experiment setup
    thresholds = [round(th, 1) for th in np.arange(start = 2.0, stop = -0.1, step = -0.2)] # thresholds on the a-VaR bounds
    if world == "feature":
        envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
    elif world == "driving":
        envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
    policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    demo_order = list(range(num_rows * num_cols))
    random.shuffle(demo_order)

    # Metrics to evaluate thresholds
    accuracies = {threshold: [] for threshold in thresholds}
    avg_bound_errors = {threshold: [] for threshold in thresholds}
    bounds = {threshold: [] for threshold in thresholds}
    true_evds = {threshold: [] for threshold in thresholds}
    num_demos = {threshold: [] for threshold in thresholds}
    pct_states = {threshold: [] for threshold in thresholds}
    policy_accuracies = {threshold: [] for threshold in thresholds}
    confidence = {threshold: 0 for threshold in thresholds}

    for i in range(num_worlds):
        env = envs[i]
        start_comp = 0
        done_with_demos = False
        for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
            D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
            demos[i].append(D)
            if debug:
                print("running BIRL with demos")
                print("demos", demos[i])
            birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
            # use MCMC to generate sequence of sampled rewards
            birl.run_mcmc(N, step_stdev, adaptive = adaptive)
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
                Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization) # compute policy loss
                policy_losses.append(Zi)

            # compute VaR bound
            N_burned = len(samples)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
            if k >= len(policy_losses):
                k = len(policy_losses) - 1
            policy_losses.sort()
            avar_bound = policy_losses[k]

            # evaluate thresholds
            for t in range(len(thresholds[start_comp:])):
                threshold = thresholds[t + start_comp]
                if avar_bound < threshold:
                    map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                    # store threshold metrics
                    accuracies[threshold].append(avar_bound >= map_evd)
                    # avg_bound_errors[threshold].append(avar_bound - map_evd)
                    # bounds[threshold].append(avar_bound)
                    # true_evds[threshold].append(map_evd)
                    # num_demos[threshold].append(M + 1)
                    # pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                    # policy_accuracies[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                    # confidence[threshold] += 1
                    if threshold == min(thresholds):
                        done_with_demos = True
                else:
                    start_comp += t
                    break
            if done_with_demos:
                break
    
    # Output results for plotting
    for threshold in thresholds:
        print("NEW THRESHOLD", threshold)
        print("Accuracy")
        if len(accuracies[threshold]) != 0:
            print(sum(accuracies[threshold]) / len(accuracies[threshold]))
        # print("Confidence")
        # print(confidence[threshold] / num_worlds)
        # print("Bound errors")
        # for abe in avg_bound_errors[threshold]:
        #     print(abe)
        # print("Policy loss bounds")
        # for apl in bounds[threshold]:
        #     print(apl)
        # print("True EVDs")
        # for tevd in true_evds[threshold]:
        #     print(tevd)
        # print("Num demos")
        # for nd in num_demos[threshold]:
        #     print(nd)
        # print("Percent states")
        # for ps in pct_states[threshold]:
        #     print(ps)
        # print("Policy accuracies")
        # for pa in policy_accuracies[threshold]:
        #     print(pa)
    print("**************************************************")
