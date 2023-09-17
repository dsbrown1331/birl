import sys
pwd = sys.path[0]
sys.path.append("/".join(pwd.split("/")[:5]) + "/")
sys.path.append("/".join(pwd.split("/")[:6]) + "/")
import random
from flask import Flask, render_template, request, jsonify
import mdp_utils
import mdp_worlds
import bayesian_irl
from mdp import FeatureMDP
import copy
from scipy.stats import norm
import numpy as np
import math
import json
import time
import re

app = Flask(__name__)

# Define the grid size (5x5)
GRID_SIZE = 5
ACTION_MAPPING = {
    "U": 0,
    "D": 1,
    "L": 2,
    "R": 3
}
feature_color = {
    1: '#D42A2F',  # Red
    2: '#2778B2',  # Blue
    3: '#339F34',  # Green
    4: '#946BBB',  # Purple
    5: '#FFFFFF'   # Goal (White)
}

teaching_option = None
selection_option = None
threshold = 0.1
num_features = None

env = None
true_optimal_policy = None
goal_state = None
given_demos = []
beta = 10
alpha = 0.95
delta = 0.05
gamma = 0.95
N = 500
step_stdev = 0.5
burn_rate = 0
skip_rate = 2
random_normalization = True
adaptive = True

final_bound = None
ground_truth_nevd = None
demo_suff = False
policy_optimality = None
policy_accuracy = None
simulation_result = {}

# Initialize the grid with random colors and empty actions (None)
colors = ["#D42A2F", "#2778B2", "#339F34", "#946BBB"]
grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
grid[23]["color"] = "white"  # goal state

given_demos = []
teaching_option = None
selection_option = None

def reset_grid():
    global grid
    grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
    grid[23]["color"] = "white"  # goal state

@app.route("/")
def index():
    return render_template("index.html", grid_size=GRID_SIZE, grid=grid, feature_color = feature_color)

@app.route("/start", methods=["POST"])
def start_simulation():
    global teaching_option, selection_option, threshold, num_features
    teaching_option = request.form.get("teaching_option")
    selection_option = request.form.get("selection_option")
    threshold = float(request.form.get("threshold_option"))
    num_features = int(request.form.get("features_option"))
    chosen_reward = request.form.get("reward_option")
    if chosen_reward != "":
        chosen_reward = re.findall(r'\d+(?:\.\d+)?', chosen_reward)
        chosen_reward = np.array([float(v) for v in chosen_reward])
        chosen_reward /= np.linalg.norm(chosen_reward)
    if teaching_option == "guided":
        grid, reward = get_environment()
    elif teaching_option == "freeform":
        grid, reward = get_environment(chosen_reward = chosen_reward)
    response = {
        "user_options": "You have chosen the {} teaching option and {} selection option for a gridworld with {} features. Let's begin!".format(teaching_option, selection_option, num_features),
        "grid": grid,
        "reward_function": reward if teaching_option == "guided" else None
    }
    return jsonify(response)

def get_environment(gridworld = True, chosen_reward = None):
    global env, goal_state, true_optimal_policy
    if gridworld:
        goal_state = random.randint(0, GRID_SIZE**2 - 1)
        terminals = [goal_state]
        env = mdp_worlds.random_feature_mdp(GRID_SIZE, GRID_SIZE, num_features - 1, terminals = terminals, chosen_reward = chosen_reward)
        true_optimal_policy = mdp_utils.get_optimal_policy(env)
        readable_grid = []
        for i in range(len(env.state_features)):
            if i % GRID_SIZE == 0:
                if i != 0:
                    readable_grid.append(readable_row)
                readable_row = []
            if i == terminals[0]:
                readable_row.append(num_features)
            else:
                arr = np.array(env.state_features[i])
                idx = int(np.argwhere(arr == 1)[0][0])
                readable_row.append(idx + 1)
        readable_grid.append(readable_row)
        reward_function = [round(float(w), 2) for w in env.feature_weights]
        print(readable_grid)
        print(reward_function)
    return readable_grid, reward_function

@app.route("/update_action", methods=["POST"])
def update_action():
    global given_demos, demo_suff, final_bound, ground_truth_nevd, policy_optimality, policy_accuracy
    square_index = int(request.form["square_index"])
    action = request.form["action"]
    grid[square_index]["action"] = action
    given_demos.append((square_index, ACTION_MAPPING[action]))
    print("Added demo", given_demos[-1])
    start = time.time()
    birl = bayesian_irl.BIRL(env, given_demos, beta)
    birl.run_mcmc(N, step_stdev, adaptive = adaptive)
    burn_idx = int(len(birl.chain) * burn_rate)
    samples = birl.chain[burn_idx::skip_rate]
    map_env = copy.deepcopy(env)
    map_env.set_rewards(birl.get_map_solution())
    map_policy = mdp_utils.get_optimal_policy(map_env)
    policy_losses = []
    for sample in samples:
        learned_env = copy.deepcopy(env)
        learned_env.set_rewards(sample)
        Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization)
        policy_losses.append(Zi)
    N_burned = len(samples)
    k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
    if k >= N_burned:
        k = N_burned - 1
    policy_losses.sort()
    avar_bound = policy_losses[k]
    end = time.time()
    print("Agent took {:02d}:{:02d}".format(int((end - start) // 60), int((end - start) % 60)))
    if avar_bound < threshold:
        demo_suff = True
        final_bound = avar_bound
        ground_truth_nevd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = 
        random_normalization)
        policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
        policy_accuracy = mdp_utils.calculate_policy_accuracy(true_optimal_policy, map_policy)
        print("Success!")
        print(mdp_utils.visualize_policy(map_policy, env))
        return jsonify({"demo_suff": True, "map_pi": map_policy, "goal": goal_state})
    else:
        if len(given_demos) == GRID_SIZE * GRID_SIZE - 1:
            final_bound = avar_bound
            ground_truth_nevd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
            policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
            policy_accuracy = mdp_utils.calculate_policy_accuracy(true_optimal_policy, map_policy)
            store_result(True)
            return jsonify({"failed": True})
        else:
            print("More demos please")
            return jsonify({"demo_suff": False})

@app.route("/end_simulation", methods=["POST"])
def end_simulation():
    reset_grid()
    global given_demos, teaching_option, selection_option
    given_demos = []
    teaching_option = None
    selection_option = None
    return "Simulation ended and data reset"

@app.route("/store_result", methods=["POST"])
def store_result(failed = False):
    global simulation_result
    simulation_result["teaching_option"] = teaching_option
    simulation_result["selection_option"] = selection_option
    simulation_result["threshold"] = threshold
    simulation_result["is_gridworld"] = isinstance(env, FeatureMDP)
    simulation_result["num_features"] = num_features  # includes augmented goal feature
    simulation_result["num_demos"] = len(given_demos)
    simulation_result["pct_states"] = len(given_demos) / (GRID_SIZE * GRID_SIZE)
    simulation_result["given_demos"] = given_demos
    simulation_result["bound"] = final_bound
    simulation_result["ground_truth_nevd"] = ground_truth_nevd
    simulation_result["good_bound"] = True if final_bound >= ground_truth_nevd else False
    simulation_result["bound_error"] = final_bound - ground_truth_nevd
    simulation_result["true_optimal_policy"] = true_optimal_policy
    simulation_result["policy_optimality"] = policy_optimality
    simulation_result["policy_accuracy"] = policy_accuracy
    simulation_result["demo_suff"] = not failed  # were demos actually sufficient or did simulation end b/c all states were shown
    simulation_result["user_evaluation"] = None if failed else request.form.get("user_response")
    with open("./results/{}-{}.json".format("failed" if failed else "success", int(time.time())), "w") as f:
        json.dump(simulation_result, f)
    return "Stored experiment result successfully"


if __name__ == "__main__":
    app.run(debug=True)
