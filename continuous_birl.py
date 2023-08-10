import continuous_utils
import numpy as np
import copy

class CONT_BIRL:
    def __init__(self, env, beta, possible_rewards, possible_policies, epsilon = 0.0001):
        self.env = copy.deepcopy(env)
        self.epsilon = epsilon
        self.beta = beta
        self.possible_rewards = possible_rewards
        self.possible_policies = possible_policies
    
    def R(self, traj, theta):
        return continuous_utils.trajreward(traj, theta, self.env.lava, traj.shape[0])
    
    def birl(self, demos):
        debug = False
        probs = []
        # counterfactuals
        # counters = self.possible_policies
        counters = []
        for pr in self.possible_rewards:
            counters.append(continuous_utils.get_human(pr, self.env.lava, type = "regular"))
        choice_set = demos + counters
        for theta in self.possible_rewards:
            n = np.exp(-self.beta * sum([self.R(demo, theta) for demo in demos]))
            d = sum([np.exp(-self.beta * self.R(demo, theta)) for demo in choice_set])**len(demos)
            probs.append(n/d)
        # for theta in self.possible_rewards:
        #     demo_rewards = [self.R(demo, theta) for demo in demos]
        #     n = np.exp(-self.beta * sum(demo_rewards))
        #     counter_rewards = np.array([self.R(demo, theta) for demo in counters])
        #     d = sum(np.exp(-self.beta * counter_rewards))**len(demos)
        #     if debug:
        #         print("For theta = {} and {} demos, ...".format(theta, len(demos)))
        #         print("Cost for demo at true theta =", self.env.feature_weights, "is", demo_rewards, "& numerator is", n)
        #         print("Costs for all policies is", counter_rewards, "& denominator is", d)
        #         print("So probability for theta = {} is {}".format(theta, n/d))
        #         print("-----")
        #     probs.append(n/d)
        #normalize belief
        Z = sum(probs)
        b = np.asarray(probs) / Z
        self.pmf = b
        self.map_sol = self.possible_rewards[np.argmax(b)]
        self.map_policy = self.possible_policies[np.argmax(b)]

    def get_map_solution(self):
        return self.map_sol
    
    def get_map_policy(self):
        return self.map_policy
    
    def get_pmf(self):
        return self.pmf