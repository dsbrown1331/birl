import random
import continuous_utils
import continuous_birl
import copy
from scipy.stats import norm
import numpy as np
import math
import sys
import argparse


if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "-b", type = str, required = True, help = "map_pi or held_out")
    parser.add_argument("--N", "-n", type = str, required = True, help = "6, 10, 20, or 50")
    args = parser.parse_args()
    stopping_condition = args.baseline
    N = int(args.N)

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 10

    # Experiment variables
    envs = [continuous_utils.random_lavaworld(tt = 1.0) for _ in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    optimality_threshold = 0.92
    max_demos = 25
    # if stopping_condition == "nevd":
    continuous_utils.initialize_parameters(n = N)
    continuous_utils.generate_random_policies(rgt = "A")
    rewards = continuous_utils.rewards

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
            if debug:
                print("Ground truth theta:", env.feature_weights)
                print("Lava is at", env.lava)
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            true_opt_policy = policies[np.where(rewards == env.feature_weights)[0][0]]
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env, generating_demo = True)
                demos[i].append(D)
                if debug:
                    print("running BIRL with additional demo starting from")
                    print("demos", demos[i][-1][0])
                birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = birl.get_map_policy()

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
                if debug:
                    print("World", i+1, ",", M+1, "demos")
                    print([d[0] for d in demos[i]])
                    print("BOUND:", avar_bound)

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = continuous_utils.calculate_expected_value_difference(env, map_env, rn = random_normalization)
                    if avar_bound < threshold:
                        if debug:
                            print("Good for bound {}; learned theta was {}".format(threshold, birl.get_map_solution()))
                        map_evd = actual
                        # store threshold metrics
                        avg_bound_errors[threshold].append(avar_bound - map_evd)
                        policy_optimalities[threshold].append(continuous_utils.calculate_policy_accuracy(env, map_env))
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
        cm100 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 100%/99%
        cm95 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 95%/96%
        cm90 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 90%/92%
        cm5 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.5
        cm4 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.4
        cm3 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.3
        cm2 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.2
        cm1 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.1

        for i in range(num_worlds):
            env = envs[i]
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            curr_map_sol = None
            patience = 0
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

                # compare policies
                if curr_map_sol == birl.get_map_solution():
                    patience += 1
                    # evaluate thresholds
                    for t in range(len(thresholds)):
                        threshold = thresholds[t]
                        actual_nevd = continuous_utils.calculate_expected_value_difference(env, map_env)
                        optimality = continuous_utils.calculate_policy_accuracy(env, map_env)
                        if patience == threshold:
                            # store metrics
                            num_demos[threshold].append(M + 1)
                            policy_optimalities[threshold].append(optimality)
                            accuracies[threshold].append(optimality >= optimality_threshold)
                            curr_map_sol = birl.get_map_solution()
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][0][0] += 1
                            else:
                                cm100[threshold][0][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][0][0] += 1
                            else:
                                cm95[threshold][0][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][0][0] += 1
                            else:
                                cm90[threshold][0][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][0][0] += 1
                            else:
                                cm5[threshold][0][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][0][0] += 1
                            else:
                                cm4[threshold][0][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][0][0] += 1
                            else:
                                cm3[threshold][0][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][0][0] += 1
                            else:
                                cm2[threshold][0][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][0][0] += 1
                            else:
                                cm1[threshold][0][1] += 1
                        else:
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][1][0] += 1
                            else:
                                cm100[threshold][1][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][1][0] += 1
                            else:
                                cm95[threshold][1][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][1][0] += 1
                            else:
                                cm90[threshold][1][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][1][0] += 1
                            else:
                                cm5[threshold][1][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][1][0] += 1
                            else:
                                cm4[threshold][1][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][1][0] += 1
                            else:
                                cm3[threshold][1][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][1][0] += 1
                            else:
                                cm2[threshold][1][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][1][0] += 1
                            else:
                                cm1[threshold][1][1] += 1
                else:
                    patience = 0
                    curr_map_sol = birl.get_map_solution()
        
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
            try:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            except ZeroDivisionError:
                print(0.0)
            print("CM100")
            print(cm100[threshold])
            print("CM95")
            print(cm95[threshold])
            print("CM90")
            print(cm90[threshold])
            print("CM5")
            print(cm5[threshold])
            print("CM4")
            print(cm4[threshold])
            print("CM3")
            print(cm3[threshold])
            print("CM2")
            print(cm2[threshold])
            print("CM1")
            print(cm1[threshold])
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
            # baseline_pis = continuous_utils.get_nonpessimal_policies(env)
            baseline_pis = [pol_set[np.random.choice(range(continuous_utils.num_rand_policies), 1)[0]] for pol_set in continuous_utils.rand_policies]
            baseline_optimality = continuous_utils.calculate_policy_accuracy(env, env, baseline_pis = baseline_pis, baseline = True)
            # print("BASELINE POLICY policy optimality {}".format(baseline_optimality))
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env)
                demos[i].append(D)
                if debug:
                    print(f"running BIRL with {M + 1} demos")
                birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies)
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                if debug:
                    print("Ground truth theta is {}, learned theta is {}".format(env.feature_weights, birl.get_map_solution()))
                map_policy = birl.get_map_policy()

                # get percent improvements
                improvements = []
                for r in range(len(rewards)):
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(rewards[r])
                    improvement = continuous_utils.calculate_percent_improvement(learned_env, map_env, baseline_pis)
                    if debug:
                        print("Percent improvement over baseline if reward was {}: {}".format(rewards[r], improvement))
                    improvements.append(improvement)
                
                # evaluate 95% confidence on lower bound of improvement
                improvements = np.nan_to_num(improvements).tolist()
                improvements.sort(reverse = True)
                N_burned = len(improvements)
                k = math.ceil(N_burned*alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= N_burned:
                    k = N_burned - 1
                bound = improvements[k]
                if debug:
                    print("Bound is", bound)
                
                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    actual = continuous_utils.calculate_percent_improvement(env, map_env, baseline_pis)
                    if debug:
                        print("Looking at threshold", threshold)
                        print("Actual improvement with ground-truth reward {}: {}".format(env.feature_weights, actual))
                    if bound > threshold:
                        # map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                        # store threshold metrics
                        avg_bound_errors[threshold].append(actual - bound)
                        policy_optimalities[threshold].append(continuous_utils.calculate_policy_accuracy(env, map_env))
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
    elif stopping_condition == "held_out":
        # Experiment setup
        thresholds = [3, 4, 5, 6, 7]

        # Metrics to evaluate stopping condition
        num_demos = {threshold: [] for threshold in thresholds}
        policy_optimalities = {threshold: [] for threshold in thresholds}
        accuracies = {threshold: [] for threshold in thresholds}
        cm100 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 100%/99%
        cm95 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 95%/96%
        cm90 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is if optimality = 90%/92%
        cm5 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.5
        cm4 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.4
        cm3 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.3
        cm2 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.2
        cm1 = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # actual positive is with nEVD threshold = 0.1

        for i in range(num_worlds):
            env = envs[i]
            policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
            demo_counter = 0
            total_demos = {threshold: [] for threshold in thresholds}
            held_out_sets = {threshold: [] for threshold in thresholds}
            for M in range(max_demos):
                D = continuous_utils.generate_optimal_demo(env)
                demo_counter += 1
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    if demo_counter % threshold == 0:
                        held_out_sets[threshold].append(D)
                        continue
                    else:
                        total_demos[threshold].append(D)
                    birl = continuous_birl.CONT_BIRL(env, beta, rewards, policies) # create BIRL environment
                    # use MCMC to generate sequence of sampled rewards
                    birl.birl(total_demos[threshold])
                    #generate evaluation policy from running BIRL
                    map_env = copy.deepcopy(env)
                    map_env.set_rewards(birl.get_map_solution())
                    map_policy = birl.get_map_policy()
                    # compare with held out set
                    if len(held_out_sets[threshold]) >= 3:
                        done = birl.get_map_solution() == env.feature_weights
                        actual_nevd = continuous_utils.calculate_expected_value_difference(env, map_env)
                        optimality = continuous_utils.calculate_policy_accuracy(env, map_env)
                        if done:
                            num_demos[threshold].append(M + 1)
                            policy_optimalities[threshold].append(optimality)
                            accuracies[threshold].append(optimality >= optimality_threshold)
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][0][0] += 1
                            else:
                                cm100[threshold][0][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][0][0] += 1
                            else:
                                cm95[threshold][0][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][0][0] += 1
                            else:
                                cm90[threshold][0][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][0][0] += 1
                            else:
                                cm5[threshold][0][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][0][0] += 1
                            else:
                                cm4[threshold][0][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][0][0] += 1
                            else:
                                cm3[threshold][0][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][0][0] += 1
                            else:
                                cm2[threshold][0][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][0][0] += 1
                            else:
                                cm1[threshold][0][1] += 1
                        else:
                            # Evaluate actual positive by optimality
                            if optimality >= 1.0:
                                cm100[threshold][1][0] += 1
                            else:
                                cm100[threshold][1][1] += 1
                            if optimality >= 0.95:
                                cm95[threshold][1][0] += 1
                            else:
                                cm95[threshold][1][1] += 1
                            if optimality >= 0.90:
                                cm90[threshold][1][0] += 1
                            else:
                                cm90[threshold][1][1] += 1
                            # Evaluate actual positive by nEVD
                            if actual_nevd < 0.5:
                                cm5[threshold][1][0] += 1
                            else:
                                cm5[threshold][1][1] += 1
                            if actual_nevd < 0.4:
                                cm4[threshold][1][0] += 1
                            else:
                                cm4[threshold][1][1] += 1
                            if actual_nevd < 0.3:
                                cm3[threshold][1][0] += 1
                            else:
                                cm3[threshold][1][1] += 1
                            if actual_nevd < 0.2:
                                cm2[threshold][1][0] += 1
                            else:
                                cm2[threshold][1][1] += 1
                            if actual_nevd < 0.1:
                                cm1[threshold][1][0] += 1
                            else:
                                cm1[threshold][1][1] += 1
        
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
            try:
                print(sum(accuracies[threshold]) / len(accuracies[threshold]))
            except ZeroDivisionError:
                print(0.0)
            print("CM100")
            print(cm100[threshold])
            print("CM95")
            print(cm95[threshold])
            print("CM90")
            print(cm90[threshold])
            print("CM5")
            print(cm5[threshold])
            print("CM4")
            print(cm4[threshold])
            print("CM3")
            print(cm3[threshold])
            print("CM2")
            print(cm2[threshold])
            print("CM1")
            print(cm1[threshold])
            print("**************************************************")
