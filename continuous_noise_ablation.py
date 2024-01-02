import random
import continuous_utils
import continuous_birl
import copy
from scipy.stats import norm
import numpy as np
import math
import time
import argparse


if __name__ == "__main__":
    start_time = time.time()
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_epsilon", "-ne", type = float, help = "0 (original), 0.05, 0.1, 0.15, 0.2, 0.25, etc...")
    parser.add_argument("--num_worlds", "-nw", type = int, help = "5 or 10")
    args = parser.parse_args()
    stopping_condition = "nevd"
    N = 6

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = 10.0 # confidence for mcmc
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = args.num_worlds

    # Experiment variables
    envs = [continuous_utils.random_lavaworld(tt = 1.0) for _ in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    optimality_threshold = 0.92
    max_demos = 25
    # if stopping_condition == "nevd":
    continuous_utils.initialize_parameters(n = N)
    continuous_utils.generate_random_policies(rgt = "A")
    rewards = continuous_utils.rewards

    # Experiment setup
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5] # thresholds on the a-VaR bounds

    # Metrics to evaluate thresholds
    num_demos = {threshold: [] for threshold in thresholds}
    confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true

    for i in range(num_worlds):
        env = envs[i]
        if debug:
            print("Ground truth theta:", env.feature_weights)
            print("Lava is at", env.lava)
        policies = [continuous_utils.get_optimal_policy(theta, env.lava) for theta in rewards]
        true_opt_policy = policies[np.where(rewards == env.feature_weights)[0][0]]
        for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
            D = continuous_utils.generate_optimal_demo(env, noise_pct = args.noise_epsilon, generating_demo = True)
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
                    # map_evd = actual
                    # store threshold metrics
                    # avg_bound_errors[threshold].append(avar_bound - map_evd)
                    # policy_optimalities[threshold].append(continuous_utils.calculate_policy_accuracy(env, map_env))
                    # accuracies[threshold].append(avar_bound >= map_evd)
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
    end_time = time.time()
    print(f"This took {round((end_time - start_time) / 60, 2)} minutes.")