import numpy as np
import mdp_utils
import copy

class PIRL:
    def __init__(self, env, epsilon = 0.001):
        self.env = copy.deepcopy(env)
        self.eps = epsilon
    
    def projection_step(self, mu, mu_bar, mu_expert):
        mu_diff_mu_bar = np.array(mu) - np.array(mu_bar)
        expert_diff_mu_bar = np.array(mu_expert) - np.array(mu_bar)
        step_size = np.dot(mu_diff_mu_bar, expert_diff_mu_bar) / np.dot(mu_diff_mu_bar, mu_diff_mu_bar)
        mu_bar_new = np.array(mu_bar) + step_size * mu_diff_mu_bar
        return mu_bar_new
    
    def get_projection_policy(self, trajectories):
        expert_feature_counts = mdp_utils.calculate_empirical_expected_fc(self.env, trajectories)
        rand_policy = mdp_utils.get_random_policy(self.env)
        mu_bar = mdp_utils.calculate_expected_fc(rand_policy, self.env, epsilon = self.eps)
        feature_weights = np.array([expert_feature_counts[i] - mu_bar[i] for i in range(self.env.num_features)])
        t = mdp_utils.two_norm_diff(expert_feature_counts, mu_bar, self.env.num_features)
        
        temp_env = copy.deepcopy(self.env)
        temp_env.set_rewards(feature_weights)
        opt_policy = mdp_utils.get_optimal_policy(temp_env, epsilon = self.eps)
        mu = mdp_utils.calculate_expected_fc(opt_policy, temp_env, epsilon = self.eps)

        t_old = 0
        count = 0
        while True:
            t_old = t
            count += 1
            mu_bar_new = self.projection_step(mu, mu_bar, expert_feature_counts)
            mu_bar = mu_bar_new
            feature_weights = np.array([expert_feature_counts[i] - mu_bar[i] for i in range(self.env.num_features)])
            t = mdp_utils.two_norm_diff(expert_feature_counts, mu_bar, self.env.num_features)
            if t < self.eps:
                print("Projection IRL has converged.")
                break
            if count > 5000:
                print("Projection IRL timed out.")
                break
            temp_env.set_rewards(feature_weights)
            opt_policy = mdp_utils.get_optimal_policy(temp_env, epsilon = self.eps)
            mu = mdp_utils.calculate_expected_fc(opt_policy, temp_env, epsilon = self.eps)
        return opt_policy