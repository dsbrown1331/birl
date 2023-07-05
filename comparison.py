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


if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    stopping_condition = "nevd"
    world = "goal"
    demo_type = "pairs" # options: pairs, trajectories
    repeat_style = sys.argv[1] # options: iid, focused

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
    N = 1112 # gets around 500 after burn and skip
    step_stdev = 0.5
    burn_rate = 0.1
    skip_rate = 2
    random_normalization = True # whether or not to normalize with random policy
    adaptive = True # whether or not to use adaptive step size
    num_worlds = 20
    max_demos = 2 * num_rows * num_cols

    if stopping_condition == "nevd": # stop learning after passing a-VaR threshold of nEVD
        # Experiment setup
        base_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05] # thresholds on the a-VaR bounds
        envs = [mdp_worlds.random_feature_mdp(num_rows, num_cols, num_features, terminals = [random.randint(0, num_rows * num_cols - 1)]) for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        if repeat_style == "iid": # repeats attained via simple randomness
            demo_states = list(range(0, num_rows * num_cols))
        elif repeat_style == "focused": # repeats attained via consistently bad demos in the corners
            demo_states = [0, num_cols - 1, num_cols * (num_rows - 1), num_cols * num_rows - 1]

        # Metrics to evaluate performance after each demo
        bounds = {i: [] for i in range(num_worlds)}
        num_demos = {i: [] for i in range(num_worlds)}
        num_unique_demos = {i: [] for i in range(num_worlds)}
        true_evds = {i: [] for i in range(num_worlds)}
        avg_bound_errors = {i: [] for i in range(num_worlds)}
        policy_optimalities = {i: [] for i in range(num_worlds)}

        accuracies = {threshold: [] for threshold in base_thresholds} # taken over all worlds at termination, NOT demos!
        # termination_metrics = {threshold: {} for threshold in thresholds}

        # Start experiment
        for i in range(num_worlds):
            env = envs[i]
            if debug:
                print("True reward:", env.feature_weights)
            thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
            if repeat_style == "focused":
                available_demos = [mdp_utils.generate_optimal_demo(env, demo_state)[0] for demo_state in demo_states]
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                try:
                    if repeat_style == "iid":
                        D = mdp_utils.generate_optimal_demo(env, random.choice(demo_states))[0]
                    else:
                        D = random.choice(available_demos)
                    demos[i].append(D)
                except IndexError:
                    pass
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos[i])
                birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev, adaptive = adaptive)
                #burn initial samples and skip every skip_rate for efficiency
                burn_indx = int(len(birl.chain) * burn_rate)
                samples = birl.chain[burn_indx::skip_rate]
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)

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
                if k >= N_burned:
                    k = N_burned - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]
                if debug:
                    print("Sample losses:", policy_losses)
                    print("a-VaR bound:", avar_bound)

                # evaluate performance
                bounds[i].append(avar_bound)
                num_demos[i].append(M + 1)
                num_unique_demos[i].append(len(set(demos[i])))
                actual = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                true_evds[i].append(actual)
                avg_bound_errors[i].append(avar_bound - actual)
                policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
                policy_optimalities[i].append(policy_optimality)

                # get threshold and accuracy information (at point of termination)
                # to_remove = []
                for threshold in thresholds:
                    if avar_bound < threshold:
                        if debug:
                            print("Passed", threshold, "threshold and actual nEVD is", actual)
                        # to_remove.append(threshold)
                        # termination_metrics[threshold] = {
                        #     "bound": avar_bound,
                        #     "num_demos": M + 1,
                        #     "num_unique_demos": len(set(demos[i])),
                        #     "true_evd": actual,
                        #     "bound_error": avar_bound - actual,
                        #     "policy_optimality": policy_optimality
                        # }
                        accuracies[threshold].append(avar_bound >= actual)
                # if len(to_remove) != 0:
                #     for threshold in to_remove:
                #         thresholds.remove(threshold)

        # Output results for plotting
        print("Number of demos")
        print(num_demos[0])
        print("Average number of unique demos across worlds for each amount of demos")
        num_unique_demos_matrix = np.array([num_unique_demos[i] for i in range(num_worlds)])
        print(list(num_unique_demos_matrix.mean(axis = 0)))
        print("Average bound across worlds for each number of demos")
        bounds_matrix = np.array([bounds[i] for i in range(num_worlds)])
        print(list(bounds_matrix.mean(axis = 0)))
        print("Average ground-truth nEVD across worlds for each number of demos")
        true_evds_matrix = np.array([true_evds[i] for i in range(num_worlds)])
        print(list(true_evds_matrix.mean(axis = 0)))
        print("Average bound error across worlds for each number of demos")
        avg_bound_errors_matrix = np.array([avg_bound_errors[i] for i in range(num_worlds)])
        print(list(avg_bound_errors_matrix.mean(axis = 0)))
        print("Average policy optimality across worlds for each number of demos")
        policy_optimalities_matrix = np.array([policy_optimalities[i] for i in range(num_worlds)])
        print(list(policy_optimalities_matrix.mean(axis = 0)))
        print("---------------------")
        print("Accuracy of each threshold at point of termination, taken over worlds")
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
        for threshold in base_thresholds:
            accuracies[threshold] = np.mean(accuracies[threshold])
            if np.isnan(accuracies[threshold]):
                accuracies[threshold] = "NO SUFFICIENCY"
        print(accuracies)
        # print("Average metric values at termination for each threshold, taken across worlds")
        # print("SOMETHING TO DO WITH TERMINATION_METRICS HERE")
        print("**************************************************")