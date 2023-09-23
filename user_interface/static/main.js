const startForm = document.getElementById("start-form"); // Get the form element
const gridContainer = document.getElementById('grid-container');
const endButton = document.getElementById('end-button'); // Get the end button
const statusUpdate = document.getElementById("status-update");
const numFeatures = 4; // Just for gridworld
var envOption = "";
const featureColor = {
    1: '#D42A2F',  // Red
    2: '#2778B2',  // Blue
    3: '#339F34',  // Green
    4: '#FFFFFF'   // Goal (White)
};
const roadColor = {
    "lane": '#666666',
    "dirt": '#955011',
};
const COLORLABEL = ['Red', 'Blue', 'Green', 'Purple', 'Goal'];
const ROADLABEL = ['Left lane', 'Middle lane', 'Right lane', 'Collision with car', 'Crash into dirt'];
var funFact = "";

startForm.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent the form submission
    const simulationOption = document.getElementById("simulation-option").value;
    const rewardOption = document.getElementById("reward-option").value;
    if ((simulationOption == "0B" || Number(simulationOption) % 2 === 0) && (rewardOption === "" || rewardOption.split(",").length !== 5)) {
        alert("Please enter in your reward function as comma-separated numbers, one for each feature in order.")
    } else {
        fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'simulation_option': document.getElementById('simulation-option').value,
                'reward_option': document.getElementById('reward-option').value
            })
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error("There was an error with /start");
            }
            return response.json();
        })
        .then(function (data) {
            funFact = data.fun_fact;
            document.getElementById("user-options").textContent = data.user_options;
            envOption = data.environment;
            if (envOption === "gridworld") {
                const grid = data.grid;
                const gridContainer = document.getElementById('grid-container');
                const gridDiv = gridContainer.querySelector('.grid');
                gridDiv.innerHTML = '';
                var squareIndex = 0;
                grid.forEach((row) => {
                    row.forEach((feature) => {
                        const square = document.createElement('div');
                        square.className = 'square';
                        square.style.backgroundColor = featureColor[feature];
                        if (feature === numFeatures) {
                            const star = document.createElement('div');
                            star.className = 'star';
                            star.innerHTML = '&starf;';
                            square.appendChild(star);
                        }
                        const indexNumber = document.createElement('div');
                        indexNumber.className = 'index-number';
                        indexNumber.textContent = squareIndex;
                        square.appendChild(indexNumber);
                        gridDiv.appendChild(square);
                        squareIndex += 1;
                    });
                });
            } else if (envOption === "driving") {
                const grid = data.grid;
                const gridContainer = document.getElementById('grid-container');
                const gridDiv = gridContainer.querySelector('.grid');
                gridDiv.innerHTML = '';
                var squareIndex = 0;
                grid.forEach((row) => {
                    row.forEach((feature) => {
                        const square = document.createElement('div');
                        square.className = 'square';
                        if (feature < 5) {
                            square.style.backgroundColor = roadColor["lane"];    
                        } else {
                            square.style.backgroundColor = roadColor["dirt"];
                        }
                        if (feature === 4) {
                            const motorist = document.createElement('div');
                            motorist.className = 'motorist';
                            motorist.innerHTML = '&#x1F697;';
                            square.appendChild(motorist);
                        }
                        const indexNumber = document.createElement('div');
                        indexNumber.className = 'index-number';
                        indexNumber.textContent = squareIndex;
                        square.appendChild(indexNumber);
                        gridDiv.appendChild(square);
                        squareIndex += 1;
                    });
                });
            }
            gridContainer.style.display = "block";
            attachEventListenersToGridSquares();
            endButton.style.display = 'block';
            var rewardFunctionString = "";
            if (envOption === "gridworld") {
                const colorLabels = COLORLABEL.slice(0, numFeatures - 1).concat(COLORLABEL.slice(-1));
                rewardFunctionString = data.reward_function.map((value, index) => {
                    const colorLabel = colorLabels[index];
                    return `<span class="${colorLabel.toLowerCase()}-text">${colorLabel}: ${value}</span>`;
                }).join(',<br>');
            } else if (envOption === "driving") {
                rewardFunctionString = data.reward_function.map((value, index) => {
                    const colorLabel = ROADLABEL[index];
                    return `<span class="${colorLabel.split(" ").join("-").toLowerCase()}-text">${colorLabel}: ${value}</span>`;
                }).join(', <br>');
            }
            var normalizationString = envOption === "driving" ? " (normalized from your submission)" : "";
    
            // Update the <span> element with the class
            document.getElementById("reward-function-vector").innerHTML = "Here is the reward function" + normalizationString + ":<br>" + "[" + rewardFunctionString + "]" + "<br>Click on a state to select it for demonstration, then enter your action.<br>Avoid states with lower reward values and go through states with higher reward values.<br><strong>You cannot undo demos</strong> so please be careful with your entries.<br><strong>Please wait for the agent to finish calculating before submitting additional demos.</strong><br><strong>If you want to stop in the middle of a simulation, just end and restart the app.</strong>";
        })
        .catch(function (error) {
            console.error("There was an error with /start:", error);
        });
    }
});

endButton.addEventListener('click', () => {
    // Send a request to end the simulation and reset data
    fetch('/end_simulation', {
        method: 'POST',
    })
    .then(response => response.text())
    .then(message => {
        // Reload the page to start a new simulation
        window.location.href = '/';
    })
    .catch(error => console.error("Error ending simulation:", error));
});

endButton.addEventListener("click", () => {
    // Clear actions
    const squares = document.querySelectorAll('.square');
    squares.forEach((square) => {
        const actionDiv = square.querySelector('.action');
        if (actionDiv) {
            square.removeChild(actionDiv);
        }
        const indexNumber = square.querySelector('.index-number');
        if (indexNumber) {
            square.removeChild(indexNumber);
        }
    });
    gridContainer.style.display = 'none';
    // Hide the undo and end buttons
    endButton.style.display = 'none';
});

function promptAction(envOption) {
    var action = "";
    if (envOption === "gridworld") {
        while (action !== "U" && action !== "D" && action !== "L" && action !== "R") {
            action = prompt("Choose an action (U, D, L, R)");
            if (action === null) {
                break;
            }
        }
    } else if (envOption === "driving") {
        while (action !== "S" && action !== "L" && action !== "R") {
            action = prompt("Choose an action (S, L, R)");
            if (action === null) {
                break;
            }
        }
    }
    return action;
}

function attachEventListenersToGridSquares() {
    const squares = document.querySelectorAll('.square');
    squares.forEach((square, index) => {
        square.addEventListener('click', () => {
            const starSquare = document.querySelector('.square .star');
            if (starSquare && starSquare.parentElement === square) {
                alert("Cannot give demonstrations for the goal state.");
            } else {
                var action = promptAction(envOption);
                if (action !== null && action !== '') {
                    const actionDiv = document.createElement('div');
                    actionDiv.classList.add('action');
                    // Set arrow symbols based on the user-selected action
                    switch (action.toUpperCase()) {
                        case 'U':
                        case 'S':
                            actionDiv.textContent = '\u2191'; // Up arrow
                            break;
                        case 'D':
                            actionDiv.textContent = '\u2193'; // Down arrow
                            break;
                        case 'L':
                            actionDiv.textContent = envOption === "gridworld" ? '\u2190' : '\u2196'; // (diag) Left arrow
                            break;
                        case 'R':
                            actionDiv.textContent = envOption === "gridworld" ? '\u2192' : '\u2197'; // (diag) Right arrow
                            break;
                        default:
                            actionDiv.textContent = action;
                    }
                    actionDiv.classList.add('arrow-icon');
                    square.appendChild(actionDiv);
                    statusUpdate.innerHTML = `<p>The agent is thinking...<br>Meanwhile, here's a fun fact for you: ${funFact}</p>`;

                    // Send the action to the server for backend processing
                    fetch('/update_action', {
                        method: 'POST',
                        body: new URLSearchParams({
                            'square_index': index,
                            'action': action
                        }),
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded'
                        }
                    })
                    .then(function (response) {
                        if (!response.ok) {
                            throw new Error("There was an error with /update_action");
                        }
                        return response.json();
                    })
                    .then (function (data) {
                        if (data.failed) {
                            displayFailedMessage();
                        } else {
                            if (data.demo_suff) {
                                displaySufficiencyMessage(data.map_pi, data.goal);
                            } else {
                                funFact = data.fun_fact;
                                displayDemoRequest();
                            }
                        }
                    })
                    .catch(function (error) {
                        console.error("There was an error with /update_action:", error);
                    });
                }
            }
        });
    });
}

function displayDemoRequest() {
    statusUpdate.innerHTML = `<p>The agent would like another demonstration.</p>`;
}

function displayFailedMessage() {
    statusUpdate.innerHTML = `<p>Unfortunately, the agent was not able to determine demonstration sufficiency (which is still good data!). This could be the agent's fault, but still be sure to keep your demonstrations as consistent as possible with one another and with the reward function.</p><p>Thank you for playing! Click End Simulation to end this session and start another round.</p>`
}

function displaySufficiencyMessage(policy, goal_state) {
    statusUpdate.innerHTML = `
        <p>&#x1F389; The agent has determined demonstration sufficiency! &#x1F389; Here is the policy it learned.<br><span style="font-size: smaller;">In case any of the arrows don't render, you can check the terminal output.</span></p>
        <p>On a scale of 1 (worst) to 5 (best), how well did the agent's learned policy match your intended policy or reward function? <strong>Please answer this, otherwise your experiment result will not be saved.</strong></p>
        <select id="user-evaluation">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
        </select>
        <button id="submit-evaluation">Submit Evaluation</button>
        <p></p>
    `;
    // Display arrows based on policy
    const squares = document.querySelectorAll('.square');
    squares.forEach((square, index) => {
        const actionDiv = square.querySelector('.action');
        if (actionDiv) {
            square.removeChild(actionDiv);
        }
        if (envOption === "driving" || index !== goal_state) {
            const arrowDiv = document.createElement('div');
            arrowDiv.classList.add('action');
            arrowDiv.style.color = 'white';
            arrowDiv.style.backgroundColor = 'transparent';
            const action = policy[index];
            switch (action) {
                case 0:
                    arrowDiv.innerHTML = '\u2191';
                    break;
                case 1:
                    arrowDiv.textContent = envOption === "gridworld" ? '\u2193' : '\u2196';
                    break;
                case 2:
                    arrowDiv.textContent = envOption === "gridworld" ? '\u2190' : '\u2197';
                    break;
                case 3:
                    arrowDiv.textContent = '\u2192';
                    break;
            }
            arrowDiv.classList.add('arrow-icon');
            square.appendChild(arrowDiv);
        }
    });
    // Disable giving more demonstrations
    disableDemonstrations();
    // Add event listener for user's response submission
    const submitEvaluationButton = document.getElementById('submit-evaluation');
    submitEvaluationButton.addEventListener('click', () => {
        const userResponse = document.getElementById('user-evaluation').value;
        fetch('/store_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'user_response': userResponse
            })
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error("There was an error with /store_result");
            }
            // After successfully storing the evaluation, display the thank you message
            statusUpdate.innerHTML = `<p>Thank you for playing! Click End Simulation to end this session and start another round.</p>`;
        })
        .catch(function (error) {
            console.error("Error storing result:", error);
        });
    });
}

// Function to disable giving more demonstrations
function disableDemonstrations() {
    const squares = document.querySelectorAll('.square');
    squares.forEach((square) => {
        var newSquare = square.cloneNode(true);
        square.parentNode.replaceChild(newSquare, square);
    });
}
