from mdp_utils import calculate_q_values, logsumexp
import numpy as np
import copy

class BIRL:
    def __init__(self, env, demos, beta, epsilon=0.0001):

        """
        Class for running and storing output of mcmc for Bayesian IRL
        env: the mdp (we ignore the reward)
        demos: list of (s,a) tuples 
        beta: the assumed boltzman rationality of the demonstrator

        """
        self.env = copy.deepcopy(env)
        self.demonstrations = demos
        self.epsilon = epsilon
        self.beta = beta

        #check to see if FeatureMDP or just plain MDP
        if hasattr(self.env, 'feature_weights'):
            self.num_mcmc_dims = len(self.env.feature_weights)
        else:
            self.num_mcmc_dims = self.env.num_states

        

    

    def calc_ll(self, hyp_reward):
        #perform hypothetical given current reward hypothesis
        self.env.set_rewards(hyp_reward)
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        for s, a in self.demonstrations:
            if (s not in self.env.terminals):  # there are no counterfactuals in a terminal state

                Z_exponents = self.beta * q_values[s]
                log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)
               
        return log_sum


    def generate_proposal(self, old_sol, stdev, normalize):
        """
        Symetric Gaussian proposal
        """
        proposal_r = old_sol + stdev * np.random.randn(len(old_sol)) 
        if normalize:
            proposal_r /= np.linalg.norm(proposal_r)
        return proposal_r


    def initial_solution(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        return np.zeros(self.num_mcmc_dims)  

    def run_mcmc(self, samples, stepsize, normalize=True):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples
        stdev = stepsize  # initial guess for standard deviation, doesn't matter too much

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce

        self.chain = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        
        cur_sol = self.initial_solution() #initial guess for MCMC

        

        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_sol = cur_sol
        for i in range(num_samples):
            # sample from proposal distribution
            prop_sol = self.generate_proposal(cur_sol, stepsize, normalize)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            if prop_ll > cur_ll:
                # accept
                self.chain[i,:] = prop_sol
                accept_cnt += 1
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_sol = prop_sol
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain[i,:] = prop_sol
                    accept_cnt += 1
                    cur_sol = prop_sol
                    cur_ll = prop_ll

                else:
                    # reject
                    self.chain[i,:] = cur_sol

        print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        # print("MAP Loglikelihood", map_ll)
        # print("MAP reward")
        # print_array_as_grid(map_sol, mdp)
        

    def get_map_solution(self):
        return self.map_sol


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''

        burn_indx = int(len(self.chain) * burn_frac)
        mean_r = np.mean(self.chain[burn_indx::skip_rate], axis=0)
        
        return mean_r
