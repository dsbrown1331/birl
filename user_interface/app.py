import sys
sys.path.append("/Users/tutrinh/Work/InterACT/birl/")
import random
from flask import Flask, render_template, request, jsonify
import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
from scipy.stats import norm
import numpy as np
import math
import json
import time
import re
import parser
fun_facts = parser.get_fun_facts()

app = Flask(__name__)

# Define the grid size (5x5)
GRID_SIZE = 5
GRIDWORLD_ACTION_MAPPING = {
    "U": 0,
    "D": 1,
    "L": 2,
    "R": 3
}
DRIVING_ACTION_MAPPING = {
    "S": 0,
    "L": 1,
    "R": 2
}
feature_color = {
    1: '#D42A2F',  # Red
    2: '#2778B2',  # Blue
    3: '#339F34',  # Green
    4: '#946BBB',  # Purple
    5: '#FFFFFF'   # Goal (White)
}
SLEEP_TIME = 20

teaching_option = None
threshold = 0.3
num_features = 4  # 3 colors, 1 star
methodology = None  # "ours", "MAP", "held_out"
environment_option = None

env = None
terminals = []
true_optimal_policy = None
goal_state = None
given_demos = []
prev_map_policy = None
patience = 3
patience_tracker = 0
held_out_set = []
held_out_tracker = 1
beta = 20
alpha = 0.95
delta = 0.05
gamma = 0.999
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
confusion_matrix = [[0, 0], [0, 0]]
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
    global teaching_option, environment_option, methodology
    simulation_option = request.form.get("simulation_option")
    if simulation_option == "0A" or simulation_option == "3":
        teaching_option = "guided"
        environment_option = "gridworld"
        methodology = "ours"
    elif simulation_option == "0B" or simulation_option == "4":
        teaching_option = "freeform"
        environment_option = "driving"
        methodology = "ours"
    elif simulation_option == "1":
        teaching_option = "guided"
        environment_option = "gridworld"
        methodology = "MAP"
    elif simulation_option == "2":
        teaching_option = "freeform"
        environment_option = "driving"
        methodology = "held_out"
    elif simulation_option == "5":
        teaching_option = "guided"
        environment_option = "gridworld"
        methodology = "held_out"
    elif simulation_option == "6":
        teaching_option = "freeform"
        environment_option = "driving"
        methodology = "MAP"
    if environment_option == "gridworld":
        grid, reward = get_environment(gridworld = True, chosen_reward = None)
    elif environment_option == "driving":
        chosen_reward = request.form.get("reward_option")
        chosen_reward = re.findall(r'-?\d+(?:\.\d+)?', chosen_reward)
        chosen_reward = np.array([float(v) for v in chosen_reward])
        chosen_reward /= np.linalg.norm(chosen_reward)
        grid, reward = get_environment(gridworld = False, chosen_reward = chosen_reward)
    response = {
        "user_options": "You have chosen simulation option {}. {}Let's begin!".format(simulation_option, "This is a practice round. " if simulation_option in ["0A", "0B"] else ""),
        "environment": environment_option,
        "grid": grid,
        "reward_function": reward if teaching_option == "guided" else [round(float(v), 2) for v in chosen_reward],
        "fun_fact": fun_facts[np.random.randint(0, len(fun_facts))]
    }
    return jsonify(response)

def get_environment(gridworld = True, chosen_reward = None):
    global env, goal_state, true_optimal_policy, terminals
    if gridworld:
        goal_state = random.randint(0, GRID_SIZE**2 - 1)
        terminals = [goal_state]
        env = mdp_worlds.random_feature_mdp(GRID_SIZE, GRID_SIZE, num_features - 1, terminals = terminals, chosen_reward = chosen_reward, user_study = True)
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
    else:
        env = mdp_worlds.random_driving_simulator(GRID_SIZE, chosen_reward = chosen_reward)
        true_optimal_policy = mdp_utils.get_optimal_policy(env)
        readable_grid = []
        for i in range(GRID_SIZE * 5):
            if i % GRID_SIZE == 0:
                if i != 0:
                    readable_grid.append(readable_row)
                readable_row = []
            arr = np.array(env.state_features[i])
            idx = int(np.argwhere(arr == 1)[0][0])
            readable_row.append(idx + 1)
        readable_grid.append(readable_row)
        reward_function = [round(float(w), 2) for w in env.feature_weights]
        print(env.motorists)
    print(readable_grid)
    print(reward_function)
    return readable_grid, reward_function

@app.route("/update_action", methods=["POST"])
def update_action():
    global given_demos, demo_suff, final_bound, ground_truth_nevd, policy_optimality, policy_accuracy, prev_map_policy, held_out_set, held_out_tracker, confusion_matrix, patience_tracker
    success = False
    skip = False
    square_index = int(request.form["square_index"])
    action = request.form["action"]
    grid[square_index]["action"] = action
    given_demo = (square_index, DRIVING_ACTION_MAPPING[action] if environment_option == "driving" else GRIDWORLD_ACTION_MAPPING[action])
    if methodology == "held_out":
        if held_out_tracker % 5 == 0:  # add every fifth demo to the validation set
            held_out_set.append(given_demo)
            skip = True
        else:
            given_demos.append(given_demo)
        held_out_tracker += 1
    else:
        given_demos.append(given_demo)
    print("Added demo", given_demos[-1])
    start = time.time()
    if not skip:  # skip is only in situation where we added a demo to the held out set
        birl = bayesian_irl.BIRL(env, given_demos, beta)
        birl.run_mcmc(N, step_stdev, adaptive = adaptive)
        burn_idx = int(len(birl.chain) * burn_rate)
        samples = birl.chain[burn_idx::skip_rate]
        map_env = copy.deepcopy(env)
        map_env.set_rewards(birl.get_map_solution())
        map_policy = mdp_utils.get_optimal_policy(map_env)
    else:
        time.sleep(SLEEP_TIME)  # to make each iteration similar in time
    if methodology == "ours":
        policy_losses = []
        for sample in samples:
            learned_env = copy.deepcopy(env)
            learned_env.set_rewards(sample)
            Zi = mdp_utils.calculate_expected_value_difference(map_policy, learned_env, birl.value_iters, rn = random_normalization)
            policy_losses.append(Zi)
        policy_losses.sort()
        N_burned = len(samples)
        k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned*alpha*(1 - alpha)) - 0.5)
        if k >= N_burned:
            k = N_burned - 1
        avar_bound = policy_losses[k]
        if avar_bound < threshold:
            success = True
    elif methodology == "MAP":
        if prev_map_policy == list(map_policy):
            patience_tracker += 1
            if patience_tracker == patience:
                success = True
        else:
            patience_tracker = 0
            prev_map_policy = list(map_policy)
        time.sleep(SLEEP_TIME)  # to make each iteration similar in time
    elif methodology == "held_out" and not skip:
        if len(held_out_set) >= 3:
            num_optimal_actions = mdp_utils.calculate_number_of_optimal_actions(map_env, map_policy, [s for s, _ in held_out_set])
            if num_optimal_actions == len(held_out_set):
                success = True
    end = time.time()
    print("Agent took {:02d}:{:02d}".format(int((end - start) // 60), int((end - start) % 60)))
    if success:
        demo_suff = True
        ground_truth_nevd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = 
        random_normalization)
        if methodology == "ours":
            final_bound = avar_bound
        if ground_truth_nevd < threshold:
            confusion_matrix[0][0] += 1
        else:
            confusion_matrix[0][1] += 1
        policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
        policy_accuracy = mdp_utils.calculate_policy_accuracy(true_optimal_policy, map_policy)
        print("Success!")
        print(mdp_utils.visualize_policy(map_policy, env))
        return jsonify({"demo_suff": True, "map_pi": map_policy, "goal": goal_state})
    else:
        if (environment_option == "driving" and (len(given_demos) + len(held_out_set)) == GRID_SIZE * GRID_SIZE) \
        or (environment_option == "gridworld" and (len(given_demos) + len(held_out_set)) == GRID_SIZE * GRID_SIZE - 1):
            if not skip:
                ground_truth_nevd = mdp_utils.calculate_expected_value_difference(map_policy, env, birl.value_iters, rn = random_normalization)
                if methodology == "ours":
                    final_bound = avar_bound
                if ground_truth_nevd < threshold:
                    confusion_matrix[1][0] += 1
                else:
                    confusion_matrix[1][1] += 1
            policy_optimality = mdp_utils.calculate_percentage_optimal_actions(map_policy, env)
            policy_accuracy = mdp_utils.calculate_policy_accuracy(true_optimal_policy, map_policy)
            store_result(True)
            return jsonify({"failed": True})
        else:
            print("More demos please")
            return jsonify({"demo_suff": False, "fun_fact": fun_facts[np.random.randint(0, len(fun_facts))]})

@app.route("/end_simulation", methods=["POST"])
def end_simulation():
    reset_grid()
    global given_demos, teaching_option, methodology, environment_option, env, terminals, true_optimal_policy, goal_state, prev_map_policy, patience_tracker, held_out_set, held_out_tracker, final_bound, ground_truth_nevd, demo_suff, policy_optimality, policy_accuracy, confusion_matrix, simulation_result
    given_demos = []
    teaching_option = None
    methodology = None
    environment_option = None
    env = None
    terminals = []
    true_optimal_policy = None
    goal_state = None
    prev_map_policy = None
    patience_tracker = 0
    held_out_set = []
    held_out_tracker = 1
    final_bound = None
    ground_truth_nevd = None
    demo_suff = False
    policy_optimality = None
    policy_accuracy = None
    confusion_matrix = [[0, 0], [0, 0]]
    simulation_result = {}
    return "Simulation ended and data reset"

@app.route("/store_result", methods=["POST"])
def store_result(failed = False):
    global simulation_result
    simulation_result["methodology"] = methodology
    simulation_result["teaching_option"] = teaching_option
    simulation_result["is_gridworld"] = environment_option == "gridworld"
    simulation_result["num_features"] = num_features if environment_option == "gridworld" else 5  # includes augmented goal feature
    simulation_result["num_demos"] = len(given_demos) + len(held_out_set)
    simulation_result["pct_states"] = (len(given_demos) + len(held_out_set)) / (GRID_SIZE * GRID_SIZE)
    simulation_result["given_demos"] = given_demos
    simulation_result["held_out_set"] = held_out_set
    simulation_result["bound"] = final_bound if methodology == "ours" else "N/A"
    simulation_result["ground_truth_nevd"] = ground_truth_nevd
    if methodology == "ours":
        simulation_result["good_bound"] = True if final_bound >= ground_truth_nevd else False
        simulation_result["bound_error"] = final_bound - ground_truth_nevd
    else:
        simulation_result["good_bound"] = "N/A"
        simulation_result["bound_error"] = "N/A"
    simulation_result["true_optimal_policy"] = true_optimal_policy
    simulation_result["policy_optimality"] = policy_optimality
    simulation_result["policy_accuracy"] = policy_accuracy
    simulation_result["confusion_matrix"] = confusion_matrix
    simulation_result["demo_suff"] = not failed  # were demos actually sufficient or did simulation end b/c all states were shown
    simulation_result["user_evaluation"] = None if failed else request.form.get("user_response")
    with open("./results/{}-{}.json".format("failed" if failed else "success", int(time.time())), "w") as f:
        json.dump(simulation_result, f)
    return "Stored experiment result successfully"


if __name__ == "__main__":
    app.run(debug=True)
