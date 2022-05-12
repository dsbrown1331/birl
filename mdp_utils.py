from mdp import MDP, FeatureMDP
from matplotlib import pyplot as plt
import time
import numpy as np
import math
import copy


def value_iteration(env, epsilon=0.0001):
    """
    TODO: speed up 
  :param env: the MDP
  :param epsilon: numerical precision for values
  :return:
  """
    n = env.num_states
    V = np.zeros(n)  # could also use np.zero(n)
    Delta = np.inf #something large to make sure we enter while loop
    
    while Delta > epsilon * (1 - env.gamma) / env.gamma:
        V_old = V.copy()
        Delta = 0
        for s in range(n):
            max_action_value = -math.inf

            for a in range(env.num_actions):
                action_value = np.dot(env.transitions[s][a], V_old)
                max_action_value = max(action_value, max_action_value)
            V[s] = env.rewards[s] + env.gamma * max_action_value
            if abs(V[s] - V_old[s]) > Delta:
                Delta = abs(V[s] - V_old[s])

    return V


def policy_evaluation(policy, env, epsilon):
    """
  Evalute the policy and compute values in each state when executing the policy in the mdp
  :param policy: the policy to evaluate in the mdp
  :param env: markov decision process where we evaluate the policy
  :param epsilon: numerical precision desired
  :return: values of policy under mdp
  """
    n = env.num_states
    V = np.zeros(n)  # could also use np.zero(n)
    Delta = 10
    
    while Delta > epsilon * (1 - env.gamma) / env.gamma:
        V_old = V.copy()
        Delta = 0
        for s in range(n):
            a = policy[s]
            policy_action_value = np.dot(env.transitions[s][a], V_old)
            V[s] = env.rewards[s] + env.gamma * policy_action_value
            if abs(V[s] - V_old[s]) > Delta:
                Delta = abs(V[s] - V_old[s])

    return V

def policy_evaluation_stochastic(env, epsilon):
    # V(s) = R(s) + gamma * sum_a T(s, a, s') sum_s'[pi(s, a, s') * V(s')]
    n = env.num_states
    num_actions = env.num_actions
    V = np.zeros(n)
    delta = 10
    while delta > epsilon * (1 - env.gamma) / env.gamma:
        V_old = V.copy()
        delta = 0
        for s in range(n):
            policy_action_value = sum([np.dot(env.transitions[s][a], V_old) for a in range(num_actions)])
            V[s] = env.rewards[s] + env.gamma * 1/num_actions * policy_action_value
            delta = max(delta, abs(V[s] - V_old[s]))
    return V


def get_optimal_policy(env, epsilon=0.0001, V=None):
    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
    n = env.num_states
    optimal_policy = []  # our game plan where we need to

    for s in range(n):
        max_action_value = -math.inf
        best_action = 0

        for a in range(env.num_actions):
            action_value = 0.0
            for s2 in range(n):  # look at all possible next states
                action_value += env.transitions[s][a][s2] * V[s2]
                # check if a is max
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = a  # direction to take
        optimal_policy.append(best_action)
    return optimal_policy


def logsumexp(x):
    max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(xi - max_x)
    return max(x) + np.log(sum_exp)


def demonstrate_entire_optimal_policy(env):
    opt_pi = get_optimal_policy(env)
    demo = []

    for state, action in enumerate(opt_pi):
        demo.append((state, action))

    return demo



<<<<<<< HEAD
def calculate_q_values(env, storage = None, V = None, epsilon = 0.0001):
=======
def calculate_q_values(env, storage = None, V=None, epsilon=0.0001):
>>>>>>> 71a527d01ba914c0901c24b620ad867586fb552c
    """
  gets q values for a markov decision process

  :param env: markov decision process
  :param epsilon: numerical precision
  :return: reurn the q values which are
  """

    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
        if storage:
            storage[env] = V
    n = env.num_states

    Q_values = np.zeros((n, env.num_actions))
    for s in range(n):
        for a in range(env.num_actions):
            Q_values[s][a] = env.rewards[s] + env.gamma * np.dot(env.transitions[s][a], V)

    return Q_values




def action_to_string(act, UP=0, DOWN=1, LEFT=2, RIGHT=3):
    if act == UP:
        return "^"
    elif act == DOWN:
        return "v"
    elif act == LEFT:
        return "<"
    elif act == RIGHT:
        return ">"
    else:
        return NotImplementedError


def visualize_trajectory(trajectory, env):
    """input: list of (s,a) tuples and mdp env
        ouput: prints to terminal string representation of trajectory"""
    states, actions = zip(*trajectory)
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in states:
                #get index
                indx = states.index(count)
                if count in env.terminals:
                    policy_row += ".\t"    
                else:    
                    policy_row += action_to_string(actions[indx]) + "\t"
            else:
                policy_row += " \t"
            count += 1
        print(policy_row)



def visualize_policy(policy, env):
    """
  prints the policy of the MDP using text arrows and uses a '.' for terminals
  """
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in env.terminals:
                policy_row += ".\t"    
            else:
                policy_row += action_to_string(policy[count]) + "\t"
            count += 1
        print(policy_row)


def print_array_as_grid(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{:.2f}\t".format(array_values[count])
            count += 1
        print(print_row)


def print_array_as_grid_raw(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{}\t".format(array_values[count])
            count += 1
        print(print_row)

def arg_max_set(values, eps=0.0001):
    # return a set of the indices that correspond to the maximum element(s) in the set of values
    # input is a list or 1-d array and eps tolerance for determining equality
    max_val = max(values)
    arg_maxes = []  # list for storing the indices to the max value(s)
    for i, v in enumerate(values):
        if abs(max_val - v) < eps:
            arg_maxes.append(i)
    return arg_maxes


def calculate_percentage_optimal_actions(pi, env, epsilon=0.0001):
    # calculate how many actions under pi are optimal under the env
    accuracy = 0.0
    # first calculate the optimal q-values under env
    q_values = calculate_q_values(env, epsilon=epsilon)
    # then check if the actions under pi are maximizing the q-values
    for state, action in enumerate(pi):
        if action in arg_max_set(q_values[state], epsilon):
            accuracy += 1  # policy action is an optimal action under env

    return accuracy / env.num_states


def calculate_expected_value_difference(eval_policy, env, storage, epsilon = 0.0001, rn = False):
    '''calculates the difference in expected returns between an optimal policy for an mdp and the eval_policy'''
    if env in storage:
        V_opt = storage[env]
    else:
        V_opt = value_iteration(env, epsilon)
    V_eval = policy_evaluation(eval_policy, env, epsilon)
    if rn:
        V_rand = policy_evaluation_stochastic(env, epsilon)
        return (np.mean(V_opt) - np.mean(V_eval)) / (np.mean(V_opt) - np.mean(V_rand))
    return np.mean(V_opt) - np.mean(V_eval)


def generate_optimal_demo(env, start_state):
    """
    Genarates a single optimal demonstration consisting of state action pairs(s,a)
    :param env: Markov decision process passed by main see (markov_decision_process.py)
    :param beta: Beta is a rationality quantification
    :param start_state: start state of demonstration
    :return:
    """
    current_state = start_state
    max_traj_length = env.num_states  #this should be sufficiently long, maybe too long...
    optimal_trajectory = []
    q_values = calculate_q_values(env)

    while (
        current_state not in env.terminals  #stop when we reach a terminal
        and len(optimal_trajectory) < max_traj_length
    ):  # need to add a trajectory length for infinite mdps

        #generate an optimal action, break ties uniformly at random
        act = np.random.choice(arg_max_set(q_values[current_state]))
        optimal_trajectory.append((current_state, act))
        probs = env.transitions[current_state][act]
        next_state = np.random.choice(env.num_states, p=probs)
        current_state = next_state

    return optimal_trajectory


def generate_boltzman_demo(env, beta, start_state):
    """
    Genarates a single boltzman rational demonstration consisting of state action pairs(s,a)
    :param env: Markov decision process passed by main see (markov_decision_process.py)
    :param beta: Beta is a rationality quantification
    :param start_state: start state of demonstration
    :return:
    """
    current_state = start_state
    max_traj = env.num_states // 2  #this should be sufficiently long, maybe too long...
    boltzman_rational_trajectory = []
    q_values = calculate_q_values(env)

    while (
        current_state not in env.terminals  #stop when we reach a terminal
        and len(boltzman_rational_trajectory) < max_traj
    ):  # need to add a trajectory length for infinite envs

        log_numerators = beta * np.array(q_values[current_state])
        boltzman_log_probs = log_numerators - logsumexp(log_numerators)
        boltzman_probability = np.exp(boltzman_log_probs)

        bolts_act = np.random.choice([0, 1, 2, 3], p=boltzman_probability)
        boltzman_rational_trajectory.append((current_state, bolts_act))
        probs = env.transitions[current_state][bolts_act]
        next_state = np.random.choice(env.num_states, p=probs)
        current_state = next_state

    return boltzman_rational_trajectory


def visualize_binary_features(env):
    #takes as input mdp_env and prints out a human readable grid of features numbered 0 to K-1, where K is number of reward
    #features. Note this method assumes binary (one-hot) features
    assert(type(env) is FeatureMDP)
    feature_values = [list(f).index(1) for f in env.state_features]
    print_array_as_grid_raw(feature_values, env)


def sample_l2_ball(k):
    #sample a vector of dimension k with l2 norm of 1
    sample = np.random.randn(k)
    return sample / np.linalg.norm(sample)