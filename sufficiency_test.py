import random
import mdp_utils
from mdp import FeatureMDP, DrivingSimulator
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math

rseed = 168
random.seed(rseed)
np.random.seed(rseed)

if __name__ == "__main__":
    stopping_condition = "nevd"

    debug = False

    # Hyperparameters
    alpha = 0.95
    delta = 0.05
    gamma = 0.95

    # MCMC hyperparameters
    beta = 10.0
    N = 50
    step_stdev = 0.5
    burn_rate = 0.1
    skip_rate = 2
    random_normalization = True

    # Experiment setup
    thresholds = [0.5]
    environments = {}
    class EnvironmentWrapper:
        def __init__(self, env, max_demos, num_states, bad_demos, good_demos):
            self.env = env
            self.max_demos = max_demos
            self.num_states = num_states
            self.bad_demos = bad_demos
            self.good_demos = good_demos

    # Environment 1
    fw = np.array([-1, -10, 10])
    white = np.array([1, 0, 0])
    red = np.array([0, 1, 0])
    star = np.array([0, 0, 1])
    sf = np.array([
        white, white, white, white, white,
        white, red, white, red, white,
        white, red, star, red, white,
        white, red, white, red, white,
        white, white, white, white, white
    ])
    env = FeatureMDP(5, 5, 4, [12], fw, sf, gamma)
    bad_demos = [0, 4, 0, 4, 0, 4, 5, 9, 15, 19, 20, 24, 20, 24, 20, 24]
    random.shuffle(bad_demos)
    good_demos = [1, 2, 3, 11, 13, 21, 22, 23]
    random.shuffle(good_demos)
    environments[1] = EnvironmentWrapper(env, 8, 25, bad_demos, good_demos)

    # Environment 2
    # feat_weights = np.array([5, 5, 10, -10, -5])
    # env = DrivingSimulator(10, [], feat_weights, [1, 3, 11, 18, 26, 28, 37, 41], [], gamma)
    # bad_demos = [0, 5, 10, 15, 20, 4, 9, 14, 19, 24]
    # random.shuffle(bad_demos)
    # good_demos = [2, 8, 12, 16, 23, 27, 33, 26, 43, 47]
    # random.shuffle(good_demos)
    # environments[2] = EnvironmentWrapper(env, 10, 50, bad_demos, good_demos)

    for i in environments:
        env_wrapper = environments[i]
        for j in range(2):
            if j == 0:
                curr_demos = env_wrapper.bad_demos
            else:
                curr_demos = env_wrapper.good_demos

            num_demos = {threshold: [] for threshold in thresholds}
            bounds = {threshold: [] for threshold in thresholds}
            demo_sufficiency = False
            demos = []

            for M in range(env_wrapper.max_demos):
                if demo_sufficiency:
                    break
                D = mdp_utils.generate_optimal_demo(env_wrapper.env, curr_demos[M])[0]
                demos.append(D)
                if debug:
                    print("running BIRL with demos")
                    print("demos", demos)
                birl = bayesian_irl.BIRL(env_wrapper.env, demos, beta)
                # use MCMC to generate sequence of sampled rewards
                birl.run_mcmc(N, step_stdev)
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
                map_env = copy.deepcopy(env_wrapper.env)
                map_env.set_rewards(birl.get_map_solution())
                map_policy = mdp_utils.get_optimal_policy(map_env)
                #debugging to visualize the learned policy
                if debug:
                    print("map policy")
                    print("MAP weights", map_env.feature_weights)
                    # mdp_utils.visualize_policy(map_policy, env)
                    print("optimal policy")
                    print("true weights", env_wrapper.env.feature_weights)
                    opt_policy = mdp_utils.get_optimal_policy(env_wrapper.env)
                    # mdp_utils.visualize_policy(opt_policy, env)
                    policy_accuracy = mdp_utils.calculate_percentage_optimal_actions(map_policy, env_wrapper.env)
                    print("policy accuracy", policy_accuracy)

                #run counterfactual policy loss calculations using eval policy
                policy_losses = []
                for sample in samples:
                    learned_env = copy.deepcopy(env_wrapper.env)
                    learned_env.set_rewards(sample)
                    # learned_policy = mdp_utils.get_optimal_policy(learned_env)
                    Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization) # compute policy loss
                    policy_losses.append(Zi)

                # compute VaR bound
                N_burned = len(samples)
                k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
                if k >= len(policy_losses):
                    k = len(policy_losses) - 1
                policy_losses.sort()
                avar_bound = policy_losses[k]

                # evaluate thresholds
                for t in range(len(thresholds)):
                    threshold = thresholds[t]
                    if avar_bound < threshold:
                        num_demos[threshold].append(M + 1)
                        bounds[threshold].append(avar_bound)
                        demo_sufficiency = True
        
            # Output results for plotting
            print(type(env_wrapper.env), "bad demos" if j == 0 else "good demos")
            for threshold in thresholds:
                print("NEW THRESHOLD", threshold)
                print("Num demos")
                for nd in num_demos[threshold]:
                    print(nd)
                print("NEVD Bounds")
                for b in bounds[threshold]:
                    print(b)
            print("**************************************************")