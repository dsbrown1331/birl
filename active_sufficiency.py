import random
import mdp_utils
import mdp_worlds
import bayesian_irl
from mdp import FeatureMDP
import copy
from scipy.stats import norm
import numpy as np
import math
import sys
import time


if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = sys.argv[1] # options: nevd, baseline
    world = sys.argv[2] # options: goal, driving

    debug = False # set to False to suppress terminal outputs

    start_time = time.time()

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95
    num_rows = 5 # 5 normal, 5 driving
    num_cols = 5 # 5 normal, 5 driving
    num_features = 4

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    N = 100 # 530 gets around 500 after burn
    step_stdev = 0.5
    burn_rate = 0.05
    skip_rate = 2
    random_normalization = True # whether or not to normalize with random policy
    adaptive = True # whether or not to use adaptive step size
    num_worlds = 1

    if stopping_condition == "nevd": # stop learning after passing a-VaR threshold
        # Experiment setup
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5] # thresholds on the a-VaR bounds
        if world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        elif world == "goal":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
            # envs = [FeatureMDP(num_rows, num_cols, 4, [], np.array([-0.2904114626557843, 0.19948602297642423, 0.9358774006220993]), np.array([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]), gamma)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]

        # Metrics to evaluate thresholds
        bounds = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        uncertain_states = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true, Pass by No Pass

        for i in range(num_worlds):
            env = envs[i]
            # print("ENVIRONMENT")
            # mdp_utils.visualize_env(env)
            demos_so_far = 0
            valid_states = np.array(list(set(range(0, env.num_states)).difference(set(env.terminals))))
            uncertain_state = np.random.choice(valid_states)
            first_state = uncertain_state
            while demos_so_far < env.num_states:
                try:
                    D = mdp_utils.generate_optimal_demo(env, uncertain_state)[0]
                    demos[i].append(D)
                except IndexError: # uncertain state is a terminal state, randomly sample another state
                    print("THIS SHOULD NOT APPEAR")
                    valid_states = np.array(list(set(valid_states).difference(set(demos[i]))))
                    D = mdp_utils.generate_optimal_demo(env, np.random.choice(valid_states))[0]
                    demos[i].append(D)
                demos_so_far += 1
                # print("Using {} demos: {}".format(demos_so_far, demos[i]))
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
                    print("environment")
                    print("state features", env.state_features)
                    print("feature weights", env.feature_weights)
                    print("map policy")
                    mdp_utils.visualize_policy(map_policy, env)
                    policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                    print("policy accuracy", policy_accuracy)

                #run counterfactual policy loss calculations using eval policy
                policy_losses = np.array([[0 for _ in range(env.num_states)]])
                for sample in samples:
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(sample)
                    policy_losses = np.append(policy_losses, mdp_utils.calculate_state_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization), axis = 0) # compute policy loss

                # compute VaR bound
                N_burned = len(samples)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= N_burned:
                    k = N_burned - 1
                policy_losses = np.delete(policy_losses, 0, axis = 0)
                policy_losses = np.sort(policy_losses, axis = 0)
                # print("POLICY LOSSES AAAAA")
                # print(policy_losses)
                avar_bound, uncertain_state = mdp_utils.find_nonterminal_uncertainties(policy_losses[k], env)
                # print("Bound = {}, uncertain in state {}".format(avar_bound, uncertain_state))

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    if demos_so_far == 1:
                        uncertain_states[threshold].append(first_state)
                    else:
                        uncertain_states[threshold].append(uncertain_state)
                    actual = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                    if avar_bound < threshold:
                        # print("SUFFICIENT ({})".format(avar_bound))
                        # print("DONE LEARNING FOR {}".format(threshold))
                        map_evd = actual
                        # store threshold metrics
                        bounds[threshold].append(avar_bound)
                        num_demos[threshold].append(demos_so_far)
                        pct_states[threshold].append(demos_so_far / (env.num_states))
                        true_evds[threshold].append(map_evd)
                        avg_bound_errors[threshold].append(avar_bound - map_evd)
                        policy_optimalities[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                        policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                        confidence[threshold].add(i)
                        accuracies[threshold].append(avar_bound >= map_evd)
                        if actual < threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                    else:
                        # print("INSUFFICIENT")
                        if actual < threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1

        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Policy loss bounds")
            for apl in bounds[threshold]:
                print(apl)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Uncertain states")
            for us in uncertain_states[threshold]:
                print(us)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("Confidence")
            print(len(confidence[threshold]) / num_worlds)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("**************************************************")
    elif stopping_condition == "map_pi": # stop learning if additional demo does not change current learned policy
        # Experiment setup
        thresholds = [1, 2, 3, 4, 5]
        if world == "feature":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features) for _ in range(num_worlds)]
        elif world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        elif world == "goal":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)

        # Metrics to evaluate stopping condition
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}

        for i in range(num_worlds):
            env = envs[i]
            curr_map_pi = [-1 for _ in range(num_rows * num_cols)]
            patience = 0
            start_comp = 0
            done_with_demos = False
            for M in range(len(demo_order)): # number of demonstrations; we want good policy without needing to see all states
                if demo_type == "pairs":
                    try:
                        D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                        demos[i].append(D)
                    except IndexError:
                        pass
                    if debug:
                        print("running BIRL with demos")
                        print("demos", demos[i])
                    birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                elif demo_type == "trajectories":
                    D = mdp_utils.generate_optimal_demo(env, demo_order[M])[:int(1/(1 - gamma))]
                    demos[i].append(D)
                    if debug:
                        print("running BIRL with demos")
                        print("demos", demos[i])
                    birl = bayesian_irl.BIRL(env, list(set([pair for traj in demos[i] for pair in traj])), beta)
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
                    patience += 1
                    # evaluate thresholds
                    for t in range(len(thresholds[start_comp:])):
                        threshold = thresholds[t + start_comp]
                        if patience == threshold:
                            # store metrics
                            num_demos[threshold].append(M + 1)
                            if demo_type == "pairs":
                                pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                            elif demo_type == "trajectories":
                                pct_states[threshold].append((M + 1) * int(1/(1 - gamma)) / (num_rows * num_cols))
                            optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                            policy_optimalities[threshold].append(optimality)
                            policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                            confidence[threshold].add(i)
                            accuracies[threshold].append(optimality >= 0.96)
                            curr_map_pi = map_policy
                            if threshold == max(thresholds):
                                done_with_demos = True
                        else:
                            start_comp += t
                            break
                else:
                    patience = 0
                    curr_map_pi = map_policy
                if done_with_demos:
                    break
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
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
            print("Confidence")
            print(len(confidence[threshold]) / num_worlds)
            print("Accuracy")
            print(sum(accuracies[threshold]) / num_worlds)
            print("**************************************************")
    elif stopping_condition == "baseline": # stop learning once learned policy is some degree better than baseline policy
        # Experiment setup
        thresholds = [round(t, 1) for t in np.arange(start = 0.0, stop = 1.1, step = 0.1)] # thresholds on the percent improvement
        if world == "driving":
            envs = [mdp_worlds.random_driving_simulator(num_rows, reward_function = "safe") for _ in range(num_worlds)]
        elif world == "goal":
            envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]

        # Metrics to evaluate thresholds
        pct_improvements = {threshold: [] for threshold in thresholds}
        num_demos = {threshold: [] for threshold in thresholds}
        pct_states = {threshold: [] for threshold in thresholds}
        true_evds = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        policy_accuracies = {threshold: [] for threshold in thresholds}
        confidence = {threshold: set() for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        uncertain_states = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true, Pass by No Pass

        for i in range(num_worlds):
            env = envs[i]
            baseline_pi = mdp_utils.get_nonpessimal_policy(env)
            baseline_evd = mdp_utils.calculate_expected_value_difference(baseline_pi, env, {}, rn = random_normalization)
            baseline_optimality = mdp_utils.calculate_percentage_optimal_actions(baseline_pi, env)
            baseline_accuracy = mdp_utils.calculate_policy_accuracy(policies[i], baseline_pi)
            demos_so_far = 0
            valid_states = np.array(list(set(range(0, env.num_states)).difference(set(env.terminals))))
            uncertain_state = np.random.choice(valid_states)
            first_state = uncertain_state
            # print("BASELINE POLICY: evd {}, policy optimality {}, and policy accuracy {}".format(baseline_evd, baseline_optimality, baseline_accuracy))
            while demos_so_far < env.num_states:
                try:
                    D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                    demos[i].append(D)
                except IndexError: # uncertain state is a terminal state, randomly sample another state
                    print("THIS SHOULD NOT APPEAR")
                    valid_states = np.array(list(set(valid_states).difference(set(demos[i]))))
                    D = mdp_utils.generate_optimal_demo(env, np.random.choice(valid_states))[0]
                    demos[i].append(D)
                demos_so_far += 1
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
                improvements = np.array([[0 for _ in range(env.num_states)]])
                for sample in samples:
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(sample)
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
                        num_demos[threshold].append(M + 1)
                        if demo_type == "pairs":
                            pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                        elif demo_type == "trajectories":
                            pct_states[threshold].append((M + 1) * int(1/(1 - gamma)) / (num_rows * num_cols))
                        true_evds[threshold].append(map_evd)
                        avg_bound_errors[threshold].append(actual - bound)
                        policy_optimalities[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                        policy_accuracies[threshold].append(mdp_utils.calculate_policy_accuracy(policies[i], map_policy))
                        confidence[threshold].add(i)
                        accuracies[threshold].append(bound <= actual)
                        if actual > threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                    else:
                        if actual > threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1
        
        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Percent Improvements")
            for pi in pct_improvements[threshold]:
                print(pi)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Percent states")
            for ps in pct_states[threshold]:
                print(ps)
            print("True EVDs")
            for tevd in true_evds[threshold]:
                print(tevd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Policy accuracies")
            for pa in policy_accuracies[threshold]:
                print(pa)
            print("Confidence")
            print(len(confidence[threshold]) / (num_worlds))
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("**************************************************")
