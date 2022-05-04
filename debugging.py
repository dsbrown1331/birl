from random import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math

if __name__ == "__main__":
    c = 100
    alpha = 0.95
    delta = 0.05
    burn_rate = 0.1
    skip_rate = 1
    beta = 10
    N = 100
    step_stdev = 0.1
    random_normalization = False

    num_worlds = 1
    envs = [mdp_worlds.random_feature_mdp(4, 4) for _ in range(num_worlds)]
    policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    demo_order = [1, 9, 12, 10, 4, 11, 5, 13, 8, 0, 14, 6, 15, 2, 3, 7]

    for M in range(16):
        print("Using {} demo{}:".format(M + 1, "" if M == 0 else "s"))
        for i in range(num_worlds):
            env = envs[i]
            opt_pi = policies[i]
            print("True policy")
            mdp_utils.visualize_policy(opt_pi, env)

            D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
            demos[i].append(D)
            print("Available demos:", demos[i])
            birl = bayesian_irl.BIRL(env, demos[i], beta)
            samples = birl.generate_samples_with_mcmc(N, step_stdev)

            map_env = copy.deepcopy(env)
            map_env.set_rewards(birl.get_map_solution())
            map_policy = mdp_utils.get_optimal_policy(map_env)
            map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, random_normalization)
            print("Learned policy")
            mdp_utils.visualize_policy(map_policy, map_env)
        print()