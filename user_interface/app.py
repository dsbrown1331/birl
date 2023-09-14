import random
from flask import Flask, render_template, request, redirect, url_for

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
colors = ["red", "blue", "green", "purple"]
grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
grid[23]["color"] = "white"  # terminal goal state

given_demos = []

def reset_grid():
    global grid
    grid = [{"color": random.choice(colors), "action": None} for _ in range(GRID_SIZE * GRID_SIZE)]
    grid[23]["color"] = "white"  # terminal goal state

@app.route("/")
def index():
    return render_template("index.html", grid_size=GRID_SIZE, grid=grid)

@app.route("/start", methods=["POST"])
def start_simulation():
    teaching_option = request.form.get("teaching_option")
    selection_option = request.form.get("selection_option")
    if teaching_option and selection_option:
        user_options = "You have chosen the {} teaching option and {} selection option. Let's begin!".format(teaching_option, selection_option)
    else:
        user_options = ""
    return render_template("index.html", user_options=user_options, grid_size=GRID_SIZE, grid=grid)

@app.route("/update_action", methods=["POST"])
def update_action():
    square_index = int(request.form["square_index"])
    action = request.form["action"]
    if square_index == 23:
        return "Cannot choose actions for the terminal goal state."
    grid[square_index]["action"] = action
    global given_demos
    given_demos.append((square_index, ACTION_MAPPING[action]))
    print("Demos given:")
    print(given_demos)
    return "Action updated successfully."

if __name__ == "__main__":
    app.run(debug=True)
