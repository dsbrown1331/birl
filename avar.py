from random import random
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math

if __name__ == "__main__":
    alphas = [0.90, 0.95, 0.99]
    for j in range(len(alphas)):
        # - MDP\R: mdp but without a defined reward function
        # - pi_eval: the policy we are evaluating, as opposed to pi*, the optimal policy. what is the
        # policy loss if we use pi_eval instead of pi*?
        # - D: set of demonstrations (in this case just 1)
        # - c: confidence in the optimality of D
        # - alpha: risk-sensitivity
        # - delta: (1 - delta) is the desired confidence level on estimate of avar.
        # higher delta => lower desired confidence; lower delta => higher desired confidence
        # - V_R^pi: (expected) value of a policy pi using the reward function R

        # Hyperparameters
        c = 100 # since demonstrations are optimal
        alpha = alphas[j]
        delta = 0.05
        burn_rate = 0.1
        skip_rate = 1
        random_normalization = False # whether or not to normalize with random policy
        num_worlds = 100
        beta = 10.0
        N = 100
        step_stdev = 0.1

        envs = [mdp_worlds.random_feature_mdp(4, 4) for _ in range(num_worlds)]
        # envs = [mdp_worlds.random_gridworld(4, 4) for _ in range(num_worlds)]
        policies = [mdp_utils.get_optimal_policy(envs[i]) for i in range(num_worlds)]
        policy_archive = [{} for _ in range(num_worlds)]
        demos = [[] for _ in range(num_worlds)]
        demo_order = [1, 9, 12, 10, 4, 11, 5, 13, 8, 0, 14, 6, 15, 2, 3, 7]
        accuracies = []
        avg_bound_errors = []
        bounds = []
        evds = []
        for M in range(1, 10): # number of demonstrations
            good_upper_bound = 0
            bound_error = []
            p_loss_bound = []
            norm_evd = []
            for i in range(num_worlds):
                env = envs[i]
                archive = policy_archive[i]
                D = mdp_utils.generate_optimal_demo(env, demo_order[M])[0]
                demos[i].append(D)
                birl = bayesian_irl.BIRL(env, demos[i], beta) # create BIRL environment
                samples = birl.generate_samples_with_mcmc(N, step_stdev)[int(burn_rate * N):] # use MCMC to generate sequence of sampled rewards
                policy_losses = []
                for sample in samples:
                    learned_env = copy.deepcopy(env)
                    sample_tuple = tuple(sample)
                    if sample_tuple in archive:
                        Zi = archive[sample_tuple]
                    else:
                        learned_env.set_rewards(sample)
                        learned_policy = mdp_utils.get_optimal_policy(learned_env)
                        Zi = mdp_utils.calculate_expected_value_difference(learned_policy, env, random_normalization) # compute policy loss
                        archive[sample_tuple] = Zi
                    policy_losses.append(Zi)
                policy_losses.sort()
                N_burned = N - int(burn_rate * N)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= N_burned:
                    k = len(policy_losses) - 1
                avar_bound = policy_losses[k]
                p_loss_bound.append(avar_bound)
                # accuracy: # of trials where upper bound > ground truth expected value difference / total # of trials
                map_env = copy.deepcopy(env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                map_evd = mdp_utils.calculate_expected_value_difference(map_policy, env, random_normalization)
                good_upper_bound += avar_bound >= map_evd
                # bound error: upper bound - EVD(eval_policy with ground truth reward)
                bound_error.append(avar_bound - map_evd)
                norm_evd.append(map_evd)
            accuracy = good_upper_bound / num_worlds
            accuracies.append(accuracy)
            avg_bound_errors.append(bound_error)
            bounds.append(p_loss_bound)
            evds.append(norm_evd)
            print("Done with {} demonstrations".format(M))
        print("Accuracies:", accuracies)
        print("Bound errors:", avg_bound_errors)
        print("Policy losses:", bounds)
        print("EVDs:", evds)
