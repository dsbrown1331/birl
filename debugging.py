import random
import continuous_birl
import continuous_utils
import copy
import numpy as np
import math
from scipy.stats import norm

if __name__ == "__main__":
    rseed = 168
    random.seed(rseed)
    np.random.seed(rseed)

    world = "lava"

    debug = False # set to False to suppress terminal outputs

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    # beta = 10.0 # confidence for mcmc
    betas = [10]
    N = continuous_utils.N
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 1

    # Experiment setup
    # thresholds = [round(th, 1) for th in np.arange(start = 0.5, stop = -0.1, step = -0.1)] # thresholds on the a-VaR bounds
    envs = [continuous_utils.random_lavaworld(tt = 0.4) for _ in range(num_worlds)]
    policies = [continuous_utils.get_optimal_policy(envs[i].feature_weights, envs[i].lava) for i in range(num_worlds)]
    # possible_rewards = [[0, 0.5], [0, 1], [0.5, 0], [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]]
    possible_rewards = np.linspace(0, 1, N)
    demos = [[] for _ in range(num_worlds)]
    max_demos = 10
    continuous_utils.generate_random_policies()

    # Metrics to evaluate thresholds
    true_evds = {num_demos + 1: [] for num_demos in range(max_demos)}
    pmfs = {num_demos + 1: 0 for num_demos in range(max_demos)}
    learned_policies = {num_demos + 1: 0 for num_demos in range(max_demos)}
    comparison_grids = {num_demos + 1: 0 for num_demos in range(max_demos)}

    beta_pmfs = []
    for beta in betas:
        for i in range(num_worlds):
            env = envs[i]
            for M in range(max_demos): # number of demonstrations; we want good policy without needing to see all states
                D = continuous_utils.generate_optimal_demo(env)
                demos[i].append(D)
                possible_policies = [continuous_utils.get_optimal_policy(pr, envs[i].lava) for pr in possible_rewards]
                birl = continuous_birl.CONT_BIRL(env, beta, possible_rewards, possible_policies)
                # use MCMC to generate sequence of sampled rewards
                birl.birl(demos[i])
                #generate evaluation policy from running BIRL
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = birl.get_map_policy()
                learned_policies[M + 1] = map_policy
                map_pmf = birl.get_pmf()
                pmfs[M + 1] = map_pmf
                beta_pmfs.append(map_pmf)

                #run counterfactual policy loss calculations using eval policy
                for j in range(len(possible_rewards)):
                    pr = possible_rewards[j]
                    learned_env = copy.deepcopy(env)
                    learned_env.set_rewards(pr)
                    Zi = continuous_utils.calculate_expected_value_difference(map_policy, learned_env, possible_policies[j], rn = random_normalization) # compute policy loss
                    true_evds[M + 1].append(Zi)

                # compute VaR bound
                # N_burned = len(policy_losses)
                # k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                # if k >= len(policy_losses):
                #     k = len(policy_losses) - 1
                # policy_losses.sort()
                # avar_bound = policy_losses[k]

                # evaluate thresholds
                # for t in range(len(thresholds[start_comp:])):
                #     threshold = thresholds[t + start_comp]
                #     if avar_bound < threshold:
                #         map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                #         # store threshold metrics
                #         accuracies[threshold].append(avar_bound >= map_evd)
                #         avg_bound_errors[threshold].append(avar_bound - map_evd)
                #         bounds[threshold].append(avar_bound)
                #         true_evds[threshold].append(map_evd)
                #         num_demos[threshold].append(M + 1)
                #         pct_states[threshold].append((M + 1) / (num_rows * num_cols))
                #         policy_accuracies[threshold].append(mdp_utils.calculate_percentage_optimal_actions(map_policy, env))
                #         confidence[threshold] += 1
                #         if threshold == min(thresholds):
                #             done_with_demos = True
                #     else:
                #         start_comp += t
                #         break
                # if done_with_demos:
                #     break
                
                grid = continuous_utils.comparison_grid(env, possible_rewards, possible_policies)
                comparison_grids[M + 1] = grid
        
        print("Environment")
        print(tuple(env.lava))
        print("True reward function")
        # print(continuous_utils.listify(env.feature_weights))
        print(env.feature_weights)
        print("True optimal policy")
        print(continuous_utils.listify(policies[i]))
        print("Possible policies")
        poss_pols = []
        for pp in possible_policies:
            poss_pols.append(continuous_utils.listify(pp))
        print(poss_pols)
        for nd in range(max_demos):
            print("Num demos", nd + 1)
            print("EVDs")
            print(true_evds[nd + 1])
            print("PMFs")
            print(list(pmfs[nd + 1]))
            print("Learned policies")
            print(continuous_utils.listify(learned_policies[nd + 1]))
            print("Comparison grid")
            print(continuous_utils.listify(comparison_grids[nd + 1]))
    print("**************************************************")
