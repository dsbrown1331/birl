import random
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define the grid size (5x5)
GRID_SIZE = 5
ACTION_MAPPING = {
    "U": 0,
    "D": 1,
    "L": 2,
    "R": 3
}

# Initialize the grid with random colors and empty actions (None)
colors = ["#D42A2F", "#2778B2", "#339F34", "#946BBB"]
grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
grid[23]["color"] = "white"  # terminal goal state

given_demos = []
teaching_option = None
selection_option = None

def reset_grid():
    global grid
    grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
    grid[23]["color"] = "white"  # terminal goal state

@app.route("/")
def index():
    return render_template("index.html", grid_size=GRID_SIZE, grid=grid)

@app.route("/start", methods=["POST"])
def start_simulation():
    global teaching_option
    global selection_option
    teaching_option = request.form.get("teaching_option")
    selection_option = request.form.get("selection_option")
    response = {
        "user_options": "You have chosen the {} teaching option and {} selection option. Let's begin!".format(teaching_option, selection_option),
        "reward_function": None if teaching_option == "freeform" else [5, 1, -1, -5, 10]
    }
    return jsonify(response)

@app.route("/update_action", methods=["POST"])
def update_action():
    square_index = int(request.form["square_index"])
    action = request.form["action"]
    if square_index == 23:
        return "Cannot give demonstrations for the terminal goal state"
    grid[square_index]["action"] = action
    global given_demos
    given_demos.append((square_index, ACTION_MAPPING[action]))
    return "Action updated successfully"

@app.route("/end_simulation", methods=["POST"])
def end_simulation():
    reset_grid()
    global given_demos
    given_demos = []
    return "Simulation ended and data reset"

if __name__ == "__main__":
    app.run(debug=True)
