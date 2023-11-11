import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math
import argparse
import time


if __name__ == "__main__":
    start_time = time.time()
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_epsilon", "-ne", type = float, help = "0 (original), 0.05, 0.1, 0.15, 0.2, 0.25")
    parser.add_argument("--num_worlds", "-nw", type = int, help = "5 or 10")
    args = parser.parse_args()
    world = "goal"
    stopping_condition = "nevd"

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
    N = 1050 # gets around 500 after burn and skip
    step_stdev = 0.5
    burn_rate = 0.05
    skip_rate = 2
    random_normalization = True # whether or not to normalize with random policy
    adaptive = True # whether or not to use adaptive step size
    num_worlds = args.num_worlds

    # Experiment setup
    thresholds = [0.3] # thresholds on the a-VaR bounds
    envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
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
    confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds} # predicted by true

    for i in range(num_worlds):
        env = envs[i]
        demo_order = list(range(num_rows * num_cols))
        random.shuffle(demo_order)
        q_values = mdp_utils.calculate_q_values(env)
        for M in range(25):
            try:
                if args.noise_epsilon == 0:  # optimal demos, one state each time, just like original
                    D = mdp_utils.generate_optimal_demo(env, demo_order[M], q_values)[0]
                else:
                    demo_probability = np.random.random()
                    demo_state = random.choice(demo_order)
                    if demo_state == env.terminals[0]:
                        raise IndexError
                    # "Epsilon-greedy" demonstrations
                    if demo_probability < args.noise_epsilon:
                        best_action = mdp_utils.generate_optimal_demo(env, demo_state, q_values)[0][1]
                        demo_action = random.choice(list(set(range(env.num_actions)) - set([best_action])))
                        D = (demo_state, demo_action)
                    else:
                        D = mdp_utils.generate_optimal_demo(env, demo_state, q_values)[0]
                demos[i].append(D)
            except IndexError:
                pass
            birl = bayesian_irl.BIRL(env, demos[i], beta)
            birl.run_mcmc(N, step_stdev, adaptive = adaptive)
            burn_indx = int(len(birl.chain) * burn_rate)
            samples = birl.chain[burn_indx::skip_rate]
            map_env = copy.deepcopy(env)
            map_env.set_rewards(birl.get_map_solution())
            map_policy = mdp_utils.get_optimal_policy(map_env)

            policy_losses = []
            for sample in samples:
                learned_env = copy.deepcopy(env)
                learned_env.set_rewards(sample)
                Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization) # compute policy loss
                policy_losses.append(Zi)
            N_burned = len(samples)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
            if k >= N_burned:
                k = N_burned - 1
            policy_losses.sort()
            avar_bound = policy_losses[k]

            for t in range(len(thresholds)):
                threshold = thresholds[t]
                actual = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                if avar_bound < threshold:
                    map_evd = actual
                    # store threshold metrics
                    bounds[threshold].append(avar_bound)
                    num_demos[threshold].append(M + 1)
                    pct_states[threshold].append((M + 1) / (num_rows * num_cols))
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
    end_time = time.time()
    print(f"This took {round((end_time - start_time) / 60, 2)} minutes.")