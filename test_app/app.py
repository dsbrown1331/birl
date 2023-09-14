import sys
pwd = sys.path[0]
sys.path.append("/".join(pwd.split("/")[:5]) + "/")
sys.path.append("/".join(pwd.split("/")[:6]) + "/")

from flask import Flask, request, render_template
import random
import mdp_utils
import mdp_worlds
import bayesian_irl
from mdp import FeatureMDP
import copy
from scipy.stats import norm
import numpy as np
import math

app = Flask(__name__)

user_options = ""

@app.route("/")
def index():
    return render_template("index.html", user_options = user_options)

@app.route("/start")
def start_simulation():
    teaching_option = request.form.get("teaching_option")
    selection_option = request.form.get("selection_option")
    user_options = "You have chosen the {} teaching option and {} selection option. Let's begin!".format(teaching_option, selection_option)
    return render_template("index.html", user_options = user_options)

@app.route('/process_input', methods=['POST'])
def process_input():
    global counter, farewell_message

    # Get user input from the form
    user_input = request.form.get('user_input')

    try:
        input_value = int(user_input)
        counter += input_value

        # Check for the farewell condition
        if counter > 69:
            farewell_message = "Goodbye! Counter exceeded 69."
    except ValueError:
        # Handle invalid input (non-integer)
        farewell_message = "Invalid input. Please enter a number."

    return render_template('index.html', counter=counter, message=farewell_message)

@app.route('/undo')
def undo():
    global counter, farewell_message

    if counter > 0:
        counter -= 1

    return render_template('index.html', counter=counter, message=farewell_message)

@app.route('/restart')
def restart():
    global counter, farewell_message
    counter = 0
    farewell_message = ""
    return render_template('index.html', counter=counter, message=farewell_message)

if __name__ == '__main__':
    app.run(debug=True)
