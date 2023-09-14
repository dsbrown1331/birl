import random
import continuous_utils
import continuous_birl
import copy
from scipy.stats import norm
import numpy as np
import math
import sys


if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = "nevd" # options: nevd, map_pi, baseline_pi
    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    N = continuous_utils.N
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 1

    # Experiment variables
    envs = [continuous_utils.random_lavaworld() for _ in range(num_worlds)]
    rewards = continuous_utils.rewards
    demos = [[] for _ in range(num_worlds)]
    max_demos = 5
    continuous_utils.generate_random_policies()

    if stopping_condition == "nevd": # stop learning after passing nEVD threshold
        # Experiment setup
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5] # thresholds on the a-VaR bounds

        # Metrics to evaluate thresholds
        num_demos = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true

        for i in range(num_worlds):
            env = envs[i]
            print("Ground truth theta:", env.feature_weights)
            print("Lava is at", env.lava)
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            print("The optimal policies start from:")
            for p in range(len(policies)):
                print(rewards[p], ":", (policies[p][0][0], policies[p][0][1]))
            true_opt_policy = policies[np.where(rewards == env.feature_weights)[0][0]]
            print("This optimal policy starts from:", (true_opt_policy[0][0], true_opt_policy[0][1]))
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env, generating_demo = True)
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = birl.get_map_policy()
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
                policy_losses = []
                for r in range(len(rewards)):
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(rewards[r])
                    Zi = continuous_utils.calculate_expected_value_difference(learned_env, map_env, rn = random_normalization) # compute policy loss
                    policy_losses.append(Zi)

                # compute VaR bound
                N_burned = len(rewards)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= N_burned:
                    k = N_burned - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]
                print("World", i+1, ",", M+1, "demos")
                print([d[0] for d in demos[i]])
                print("BOUND:", avar_bound)
                print("\n")

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = continuous_utils.calculate_expected_value_difference(env, map_env, rn = random_normalization)
                    if avar_bound < threshold:
                        print("Good for bound {}; learned theta was {}".format(threshold, birl.get_map_solution()))
                        map_evd = actual
                        # store threshold metrics
                        avg_bound_errors[threshold].append(avar_bound - map_evd)
                        policy_optimalities[threshold].append(continuous_utils.calculate_policy_accuracy(env, map_policy, opt_pi = true_opt_policy))
                        accuracies[threshold].append(avar_bound >= map_evd)
                        num_demos[threshold].append(M + 1)
                        if actual < threshold:
                            confusion_matrices[threshold][0][0] += 1
                        else:
                            confusion_matrices[threshold][0][1] += 1
                    else:
                        if actual < threshold:
                            confusion_matrices[threshold][1][0] += 1
                        else:
                            confusion_matrices[threshold][1][1] += 1

        # Output results for plotting
        for threshold in thresholds:
            print("NEW THRESHOLD", threshold)
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
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

        # Metrics to evaluate stopping condition
        num_demos = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}

        for i in range(num_worlds):
            env = envs[i]
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            true_opt_policy = continuous_utils.get_optimal_policy(env.feature_weights, env.lava)
            curr_map_pi = np.zeros(true_opt_policy.shape)
            patience = 0
            start_comp = 0
            done_with_demos = False
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env)
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = birl.get_map_policy()
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
                policy_match = continuous_utils.calculate_policy_accuracy(curr_map_pi, map_policy)
                if policy_match == 1.0:
                    patience += 1
                    # evaluate thresholds
                    for t in range(len(thresholds[start_comp:])):
                        threshold = thresholds[t + start_comp]
                        if patience == threshold:
                            # store metrics
                            optimality = continuous_utils.calculate_policy_accuracy(map_policy, true_opt_policy)
                            policy_optimalities[threshold].append(optimality)
                            accuracies[threshold].append(optimality >= 1.0)
                            num_demos[threshold].append(M + 1)
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
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Accuracy")
            print(sum(accuracies[threshold]) / num_worlds)
            print("**************************************************")
    elif stopping_condition == "baseline_pi": # stop learning once learned policy is some degree better than baseline policy
        # Experiment setup
        thresholds = [round(t, 1) for t in np.arange(start = 0.0, stop = 1.1, step = 0.1)] # thresholds on the percent improvement

        # Metrics to evaluate thresholds
        num_demos = {threshold: [] for threshold in thresholds}
        avg_bound_errors = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true

        for i in range(num_worlds):
            env = envs[i]
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            true_opt_policy = continuous_utils.get_optimal_policy(env.feature_weights, env.lava)
            baseline_pi = continuous_utils.get_nonpessimal_policy(env)
            baseline_evd = continuous_utils.calculate_expected_value_difference(baseline_pi, env, true_opt_policy, rn = random_normalization)
            baseline_optimality = continuous_utils.calculate_policy_accuracy(baseline_pi, true_opt_policy)
            print("BASELINE POLICY: evd {}, policy optimality {}, and policy accuracy {}".format(baseline_evd, baseline_optimality, 69))
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env)
                demos[i].append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies)
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = birl.get_map_policy()
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
                for r in range(len(rewards)):
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(rewards[r])
                    improvement = continuous_utils.calculate_percent_improvement(learned_env, baseline_pi, map_policy)
                    improvements.append(improvement)
                
                # evaluate 95% confidence on lower bound of improvement
                improvements = np.nan_to_num(improvements).tolist()
                improvements.sort(reverse = True)
                N_burned = len(improvements)
                k = math.ceil(N_burned*alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= N_burned:
                    k = N_burned - 1
                bound = improvements[k]
                
                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = continuous_utils.calculate_percent_improvement(env, baseline_pi, map_policy, actual = "True")
                    if bound > threshold:
                        # map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                        # store threshold metrics
                        avg_bound_errors[threshold].append(actual - bound)
                        policy_optimalities[threshold].append(continuous_utils.calculate_policy_accuracy(map_policy, true_opt_policy))
                        accuracies[threshold].append(bound <= actual)
                        num_demos[threshold].append(M + 1)
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
            print("Num demos")
            for nd in num_demos[threshold]:
                print(nd)
            print("Bound errors")
            for abe in avg_bound_errors[threshold]:
                print(abe)
            print("Policy optimalities")
            for po in policy_optimalities[threshold]:
                print(po)
            print("Accuracy")
            if len(accuracies[threshold]) != 0:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            else:
                print(0.0)
            print("Confusion matrices")
            print(confusion_matrices[threshold])
        print("**************************************************")
