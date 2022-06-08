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
    stopping_condition = sys.argv[1]

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    num_rows = 5 # 4 normal, 5 driving
    num_cols = 5 # 4 normal, 5 driving
    num_features = 3

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    N = 450
    step_stdev = 0.3
    burn_rate = 0.05
    skip_rate = 2
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 20

    if stopping_condition == "avar": # stop learning after passing a-VaR threshold
        # Experiment setup
        thresholds = [round(th, 1) for th in np.arange(start = 2.0, stop = -0.1, step = -0.2)] # thresholds on the a-VaR bounds
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

        for i in range(num_worlds):
            env = envs[i]
            start_comp = 0
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
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
                        avg_bound_errors[threshold].append(avar_bound - map_evd)
                        bounds[threshold].append(avar_bound)
                        true_evds[threshold].append(map_evd)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                        policy_accuracies[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                    else:
                        start_comp += t
                        break
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
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
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
        print("**************************************************")
    elif stopping_condition == "wfcb": # stop learning after avar bound < worst-case feature-count bound
        # Experiment setup
        envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate stopping condition
        accuracy = 0
        avg_bound_errors = []
        bounds = []
        true_evds = []
        num_demos = []
        pct_states = []
        policy_accuracies = []

        for i in range(num_worlds):
            env = envs[i]
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
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

                # compute VaR bound
                N_burned = len(samples)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= len(policy_losses):
                    k = len(policy_losses) - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]

                # compare bounds
                wfcb = mdp_utils.calculate_wfcb(map_policy, env, [demos[i]])
                print("WFCB", wfcb)
                if avar_bound < wfcb:
                    map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                    # store metrics
                    accuracy += avar_bound >= map_evd
                    avg_bound_errors.append(avar_bound - map_evd)
                    bounds.append(avar_bound)
                    true_evds.append(map_evd)
                    num_demos.append(M + 1)
                    pct_states.append((M + 1) / (num_rows * num_cols))
                    policy_accuracies.append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                    break
        
        # Output results for plotting
        print("Accuracy")
        print(accuracy / num_worlds)
        print("Bound errors")
        for abe in avg_bound_errors:
            print(abe)
        print("Policy loss bounds")
        for apl in bounds:
            print(apl)
        print("True EVDs")
        for tevd in true_evds:
            print(tevd)
        print("Num demos")
        for nd in num_demos:
            print(nd)
        print("Percent states")
        for ps in pct_states:
            print(ps)
        print("Policy accuracies")
        for pa in policy_accuracies:
            print(pa)
        print("**************************************************")
    elif stopping_condition == "wfcb_threshold": # stop learning after passing WFCB threshold
        # Experiment setup
        thresholds = np.arange(start = 10, stop = -1, step = -1) # thresholds on the WFCB bounds
        envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate thresholds
        accuracies = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        wfcb_bounds = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}

        for i in range(num_worlds):
            env = envs[i]
            start_comp = 0
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
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

                # compute WFCB bound
                wfcb = mdp_utils.calculate_wfcb(map_policy, env, [demos[i]])

                # evaluate thresholds
                for t in range(len(thresholds[start_comp:])):
                    threshold = thresholds[t + start_comp]
                    if wfcb < threshold:
                        map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                        # store threshold metrics
                        accuracies[threshold].append(wfcb >= map_evd)
                        avg_bound_errors[threshold].append(wfcb - map_evd)
                        wfcb_bounds[threshold].append(wfcb)
                        true_evds[threshold].append(map_evd)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                        policy_accuracies[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                    else:
                        start_comp += t
                        break
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("WFCBs")
            for wfcb in wfcb_bounds[threshold]:
                print(wfcb)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
        print("**************************************************")
    elif stopping_condition == "map_pi": # stop learning if additional demo does not change current learned policy
        # Experiment setup
        envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate stopping condition
        accuracy = 0
        avg_bound_errors = []
        bounds = []
        true_evds = []
        num_demos = []
        pct_states = []
        policy_accuracies = []

        for i in range(num_worlds):
            env = envs[i]
            curr_map_pi = [-1 for _ in range(num_rows * num_cols)]
            for M in range(0, len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
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

                # compute VaR bound
                N_burned = len(samples)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= len(policy_losses):
                    k = len(policy_losses) - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]

                # compare policies
                policy_match = mdp_utils.calculate_policy_accuracy(curr_map_pi, map_policy)
                if policy_match == 1.0:
                    map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                    # store metrics
                    accuracy += avar_bound >= map_evd
                    avg_bound_errors.append(avar_bound - map_evd)
                    bounds.append(avar_bound)
                    true_evds.append(map_evd)
                    num_demos.append(M + 1)
                    pct_states.append((M + 1) / (num_rows * num_cols))
                    policy_accuracies.append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                    break
                else:
                    curr_map_pi = map_policy
        
        # Output results for plotting
        print("Accuracy")
        print(accuracy / num_worlds)
        print("Bound errors")
        for abe in avg_bound_errors:
            print(abe)
        print("Policy loss bounds")
        for apl in bounds:
            print(apl)
        print("True EVDs")
        for tevd in true_evds:
            print(tevd)
        print("Num demos")
        for nd in num_demos:
            print(nd)
        print("Percent states")
        for ps in pct_states:
            print(ps)
        print("Policy accuracies")
        for pa in policy_accuracies:
            print(pa)
        print("**************************************************")
