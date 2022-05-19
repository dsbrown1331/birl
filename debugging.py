import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math

rseed = 168
random.seed(rseed)
np.random.seed(rseed)

if __name__ == "__main__":
    # Hyperparameters
    max_num_demos = 9 # maximum number of demos to give agent, start with 1 demo and then work up to max_num_demos
    alphas = [0.90, 0.95, 0.99]
    delta = 0.05
    num_rows = 5 # 4 normal, 5 driving
    num_cols = 4
    num_features = 3

    # MCMC hyperparameters
    beta = 10.0  # confidence for mcmc
    N = 10
    step_stdev = 0.3
    burn_rate = 0.05
    skip_rate = 1
    random_normalization = True # whether or not to normalize with random policy
    num_worlds = 1

    envs = [mdp_worlds.random_driving_simulator(5) for _ in range(num_worlds)]
    policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
    demos = [[] for _ in range(num_worlds)]
    # demo_order = [0, 6, 12, 18, 24] + [25, 31, 37, 43, 49] + [50, 56, 62, 68, 74]
    demo_order = list(range(75))
    random.shuffle(demo_order)

    for M in range(len(demo_order)):
        print("Using {} demo{}:".format(M + 1, "" if M == 0 else "s"))
        for i in range(num_worlds):
            env = envs[i]
            opt_pi = policies[i]
            print("True policy")
            # mdp_utils.visualize_policy(opt_pi, env)
            print(opt_pi)

            D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
            demos[i].append(D)
            print("Available demos:", demos[i])
            birl = bayesian_irl.BIRL(env, demos[i], beta)
            samples = birl.generate_samples_with_mcmc(N, step_stdev)

            map_env = copy.deepcopy(env)
            map_env.set_rewards(birl.get_map_solution())
            map_policy = mdp_utils.get_optimal_policy(map_env)
            map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
            print("Learned policy")
            # mdp_utils.visualize_policy(map_policy, map_env)
            print(map_policy)

            sames = 0
            for j in range(len(opt_pi)):
                if opt_pi[j] == map_policy[j]:
                    sames += 1
            print("Policy accuracy:", round(sames / len(opt_pi) * 100, 2))            
        print()