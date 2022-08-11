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
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = sys.argv[1] # options: avar, wfcb, wfcb_threshold, map_pi, baseline_pi
    world = sys.argv[2] # options: feature, driving

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
    N = 555 # gets around 500 after burn and skip
    step_stdev = 0.5
    burn_rate = 0.10
    skip_rate = 1
    random_normalization = True # whether or not to normalize with random policy
    adaptive = True # whether or not to use adaptive step size
    num_worlds = 20

    if stopping_condition == "avar": # stop learning after passing a-VaR threshold
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
                        confidence[threshold] += 1
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
            print("Confidence")
            print(confidence[threshold] / num_worlds)
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
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
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
                    policy_accuracies.append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
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
        thresholds = [round(t, 2) for t in np.arange(start = 1.0, stop = -0.01, step = 0.79)] # thresholds on the WFCB bounds
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
        wfcb_bounds = {threshold: [] for threshold in thresholds}
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
                D = mdp_utils.generate_optimal_demo(env, demo_order[M])[:int(1/(1 - gamma))]
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = bayesian_irl.BIRL(env, list(set([pair for traj in demos[i] for pair in traj])), beta) # create BIRL environment
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

                # compute WFCB bound
                wfcb = mdp_utils.calculate_wfcb(map_policy, env, demos[i])

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
                        confidence[threshold] += 1
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
            print("Confidence")
            print(confidence[threshold] / num_worlds)
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
                print(nd * int(1/(1 - gamma)))
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps * int(1/(1 - gamma)))
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
        print("**************************************************")
    elif stopping_condition == "map_pi": # stop learning if additional demo does not change current learned policy
        # Experiment setup
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
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

                # compare policies
                policy_match = mdp_utils.calculate_policy_accuracy(curr_map_pi, map_policy)
                if policy_match == 1.0:
                    map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
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
                    # store metrics
                    accuracy += avar_bound >= map_evd
                    avg_bound_errors.append(avar_bound - map_evd)
                    bounds.append(avar_bound)
                    true_evds.append(map_evd)
                    num_demos.append(M + 1)
                    pct_states.append((M + 1) / (num_rows * num_cols))
                    policy_accuracies.append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
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
    elif stopping_condition == "baseline_pi": # stop learning once learned policy is some degree better than baseline policy
        # Experiment setup
        thresholds = [round(t, 1) for t in np.arange(start = 0.0, stop = 1.0, step = 0.1)] + [round(t, 1) for t in np.arange(start = 1.0, stop = 5.5, step = 0.5)] # thresholds on the percent improvement
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate thresholds
        pct_improvements = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true

        for i in range(num_worlds):
            # print("@@@ Evaluation world {} @@@".format(i))
            env = envs[i]
            baseline_pi = mdp_utils.get_nonpessimal_policy(env)
            baseline_evd = mdp_utils.calculate_expected_value_difference(baseline_pi, env, {}, rn = random_normalization)
            baseline_optimality = mdp_utils.calculate_percentage_optimal_actions(baseline_pi, env)
            baseline_accuracy = mdp_utils.calculate_policy_accuracy(policies[i], baseline_pi)
            print("BASELINE POLICY: evd {}, policy optimality {}, and policy accuracy {}".format(baseline_evd, baseline_optimality, baseline_accuracy))
            # start_comp = 0
            # done_with_demos = False
            for M in range(len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
                # print("Using {} demos".format(M + 1))
                D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                demos[i].append(D)
                if debug:
                    print("Running BIRL with {} demos:".format(M + 1), demos[i])
                birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev, adaptive = adaptive)
                #burn initial samples and skip every skip_rate for efficiency
                burn_indx = int(len(birl.chain) * burn_rate)
                samples = birl.chain[burn_indx::skip_rate]
                #check if MCMC seems to be mixing properly
                if debug:
                    if birl.accept_rate > 0.7:
                        msg = ", too high, probably need to increase stdev"
                    elif birl.accept_rate < 0.2:
                        msg = ", too low, probably need to decrease stdev"
                    else:
                        msg = ""
                    print("accept rate: " + str(birl.accept_rate) + msg) #good to tune number of samples and stepsize to have this around 50%
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                #debugging to visualize the learned policy
                if debug:
                    print("True weights", env.feature_weights)
                    print("True policy")
                    # mdp_utils.visualize_policy(policies[i], env)
                    print("MAP weights", map_env.feature_weights)
                    print("MAP policy")
                    # mdp_utils.visualize_policy(map_policy, map_env)
                    policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                    print("Policy optimality:", policy_optimality)
                    policy_accuracy = mdp_utils.calculate_policy_accuracy(policies[i], map_policy)
                    print("Policy accuracy:", policy_accuracy)

                # get percent improvements
                improvements = []
                for sample in samples:
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(sample)
                    # print("Sample reward:", sample)
                    vb, ve, improvement = mdp_utils.calculate_percent_improvement(learned_env, baseline_pi, map_policy)
                    # print("V base: {}, V evail: {}".format(vb, ve))
                    improvements.append(improvement)
                # print("Percent improvement: {}".format(improvement))
                # print("Mean V_base: {}, mean V_eval: {}, percent improvement: {}".format(V_base, V_eval, improvement))

                # evaluate 95% confidence on lower bound of improvement
                improvements = np.nan_to_num(improvements).tolist()
                improvements.sort(reverse = True)
                N_burned = len(samples)
                # k = math.ceil(N_burned*(1 - alpha) + norm.ppf(1 - delta) * np.sqrt(N_burned*(1 - alpha)*alpha) - 0.5)
                k = math.ceil(N_burned*alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                # print("Improvements:", improvements)
                # print("Bound {}: {} of {}".format(improvements[k], k, N_burned))
                if k >= N_burned:
                    k = N_burned - 1
                bound = improvements[k]
                if debug:
                    print("VaR bound:", bound)
                
                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    _, _, actual = mdp_utils.calculate_percent_improvement(env, baseline_pi, map_policy)
                    if bound > threshold:
                        # print("Comparing {} with threshold {}, passed".format(improvement, threshold))
                        map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                        # store threshold metrics
                        pct_improvements[threshold].append(bound)
                        true_evds[threshold].append(map_evd)
                        num_demos[threshold].append(M + 1)
                        pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                        policy_optimalities[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                        policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                        confidence[threshold].add(i)
                        if actual > threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                        # if threshold == max(thresholds):
                        #     done_with_demos = True
                    else:
                        if actual > threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1
                        # start_comp += t
                        # print("Comparing {} with threshold {}, did not pass".format(improvement, threshold))
                        # break
                # if done_with_demos:
                    # break
            # print("\n\n\n")
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Percent Improvements")
            for pi in pct_improvements[threshold]:
                print(pi)
            print("Confidence")
            print(len(confidence[threshold]) / (num_worlds))
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("**************************************************")
