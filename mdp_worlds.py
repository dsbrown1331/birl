from mdp import FeatureMDP, MDP
import numpy as np
import mdp_utils




def gen_simple_world():
    #four features, red (-1), blue (+5), white (0), yellow (+1)
    r = [1,0,0]
    b = [0,1,0]
    w = [0,0,1]
    gamma = 0.9
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, w, 
                      w, r, w,
                      w, w, w]
    feature_weights = [-1.0, 1.0, 0.0] #red feature has weight -1 and blue feature has weight +1
    
    noise = 0.0 #no noise in transitions
    env = FeatureMDP(3,3,[0],feature_weights, state_features, gamma, noise)
    return env


def random_gridworld(rows, columns):
    """
    Randomly chooses rewards, no terminal, noisy transitions
    """
    random_mdp = MDP(rows, columns, [], np.random.randn(rows * columns), gamma=0.95, noise = 0.1)
    return random_mdp


def random_feature_mdp(rows, columns, num_features):
    """
    e.g. if each cell in a 3x3 non-terminal grid world has a color: red or white, and red states have as reward of -1
    and white states have a reward of +1, then this could be created by having
    feature_weights = np.array([-1.0, +1.0])
    r = np.array([1,0]) #features for red
    w = np.array([0,1]) #features for white
    state_features = [w, w, w,
                        r, r, w
                        r, w, w]
    mdp = FeatureMDP(3,3,[], reward_weights, reward_features, 0.95, noise = 0)

    (self, num_rows, num_cols, terminals, feature_weights, state_features, gamma, noise = 0.0)
    """
    # weights = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # feature_weights = np.random.choice(weights, size = 3, replace = True)
    feature_weights = mdp_utils.sample_l2_ball(num_features)
    # r = np.array([1, 0, 0])
    # y = np.array([0, 1, 0])
    # b = np.array([0, 0, 1])
    # g = np.array([0, 0, 0, 1])
    # features = [r, y, b]
    #create one-hot binary feature vectors
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]
    #randomly select features for each state in mdp
    state_features = [features[np.random.randint(num_features)] for _ in range(rows * columns)]
    mdp = FeatureMDP(rows, columns, [], feature_weights, state_features, gamma = 0.95)
    return mdp



if __name__ =="__main__":

    '''Here's a simple example of how to use the FeatureMDP class'''


    
    #three features, red (-1), blue (+1), white (0)
    r = [1,0]
    b = [0,1]
    w = [0,0]
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, 
                      w, w]
    feature_weights = [-1.0, 1.0] #red feature has weight -1 and blue feature has weight +1
    gamma = 0.5
    noise = 0.0
    eps = 0.0001
    mdp = FeatureMDP(2,2,[0],feature_weights, state_features, gamma, noise)
    
    opt_pi = value_iteration.get_Optimal_Policy(mdp, eps, slow=True)
    print("optimal policy for weights [-1,1]")
    value_iteration.print_policy_pretty(opt_pi, mdp)
    print("rewards for weights [-1,1] as a grid")
    value_iteration.print_array_as_grid(mdp.reward, mdp)
    print()

    #now let's try and set the weights to a different value
    new_weights = [+1, -1] #now red is good and blue is bad
    mdp.set_reward(new_weights)
    opt_pi = value_iteration.get_Optimal_Policy(mdp, eps, slow=True)
    print("optimal policy for weights", new_weights)
    value_iteration.print_policy_pretty(opt_pi, mdp)
    print("rewards for weights", new_weights,  "as a grid")
    value_iteration.print_array_as_grid(mdp.reward, mdp)


    eval_policy1 = [0,0,3,0]
    eval_policy2 = [2,2,2,2]
    print(value_iteration.calculate_value_difference(eval_policy1, mdp, eps))
    print(value_iteration.calculate_value_difference(eval_policy2, mdp, eps))

    
    beta = 1.0
    samples = 1000
    stepsize = 0.1
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, w, w, w, 
                      w, r, w, w, w,
                      w, r, w, w, w,
                      w, r, w, w, w,
                      w, w, w, w, w]
    feature_weights = [-1.0, 1.0] #red feature has weight -1 and blue feature has weight +1
    gamma = 0.5
    noise = 0.0
    eps = 0.0001
    mdp = FeatureMDP(5, 5,[0],feature_weights, state_features, gamma, noise)
    print("rewards", mdp.reward)
    print(type(mdp))
    print(isinstance(mdp, FeatureMDP))
    opt_pi = value_iteration.get_Optimal_Policy(mdp, eps)
    value_iteration.print_policy_pretty(opt_pi, mdp)
    demonstrations = value_iteration.demonstrate_optimal_policy(mdp)
    map_solution, mean_r, mean_beta = value_iteration.run_mcmc(mdp, demonstrations, samples, stepsize, eps)
    print("map", map_solution)
    print("mean_r", mean_r)
    print("mean beta", mean_beta)
    mdp_copy = copy.deepcopy(mdp)
    mdp_copy.set_reward(mean_r)
    learned_opt_pi = value_iteration.get_Optimal_Policy(mdp_copy, eps)
    value_iteration.print_policy_pretty(learned_opt_pi, mdp_copy)
