const startForm = document.getElementById("start-form"); // Get the form element
const gridContainer = document.getElementById('grid-container');
const endButton = document.getElementById('end-button'); // Get the end button
const statusUpdate = document.getElementById("status-update");
const featureColor = {
    1: '#D42A2F',  // Red
    2: '#2778B2',  // Blue
    3: '#339F34',  // Green
    4: '#946BBB',  // Purple
    5: '#FFFFFF'   // Goal (White)
};
const roadColor = {
    "lane": '#666666',
    "dirt": '#955011',
};
const COLORLABEL = ['Red', 'Blue', 'Green', 'Purple', 'Goal'];
const ROADLABEL = ['Left lane', 'Middle lane', 'Right lane', 'Collision with car', 'Crash into dirt'];
var firstSelection = true;
var requestedState = null;
var funFact = "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible. Honey's low water content and acidic pH create an environment that is inhospitable to most bacteria and microorganisms, allowing it to remain preserved for incredibly long periods of time.";

startForm.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent the form submission
    const teachingOption = document.getElementById('teaching-option').value;
    const rewardOption = document.getElementById('reward-option').value;
    const featuresOption = document.getElementById('features-option').value;
    if (teachingOption === "freeform" && (rewardOption === "" || rewardOption.split(",").length !== Number(featuresOption))) {
        alert("Please enter in your reward function as comma-separated numbers, one for each feature.")
    } else {
        fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'environment_option': document.getElementById('environment-option').value,
                'teaching_option': document.getElementById('teaching-option').value,
                'selection_option': document.getElementById('selection-option').value,
                'threshold_option': document.getElementById('threshold-option').value,
                'features_option': document.getElementById('features-option').value,
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
            document.getElementById("user-options").textContent = data.user_options;
            const envOption = document.getElementById('environment-option').value;
            if (envOption === "gridworld") {
                const grid = data.grid;
                const gridContainer = document.getElementById('grid-container');
                const gridDiv = gridContainer.querySelector('.grid');
                const num_features = parseInt(document.getElementById('features-option').value);
                gridDiv.innerHTML = '';
                var squareIndex = 0;
                grid.forEach((row) => {
                    row.forEach((feature) => {
                        const square = document.createElement('div');
                        square.className = 'square';
                        if (feature === num_features) {
                            square.style.backgroundColor = featureColor[5];    
                        } else {
                            square.style.backgroundColor = featureColor[feature];
                        }
                        if (feature === num_features) {
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
                const num_features = parseInt(document.getElementById('features-option').value);
                const colorLabels = COLORLABEL.slice(0, num_features - 1).concat(COLORLABEL.slice(-1));
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
            const teachingOption = document.getElementById('teaching-option').value;
            var normalizationString = teachingOption === "freeform" ? "(normalized from your submission) " : "";
    
            // Update the <span> element with the class
            document.getElementById("reward-function-vector").innerHTML = "Here is the reward function " + normalizationString + ":<br>" + "[" + rewardFunctionString + "]" + "<br>Click on a state to select it for demonstration, then enter your action.<br>Avoid states with lower reward values and go through states with higher reward values.<br><strong>You cannot undo demos</strong> so please be careful with your entries and only type in letters that are prompted.<br><strong>Please wait for the agent to finish calculating before submitting additional demos.</strong>";
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

function attachEventListenersToGridSquares() {
    const squares = document.querySelectorAll('.square');
    squares.forEach((square, index) => {
        square.addEventListener('click', () => {
            const starSquare = document.querySelector('.square .star');
            const envOption = document.getElementById('environment-option').value;
            const selectionOption = document.getElementById('selection-option').value;
            if (starSquare && starSquare.parentElement === square) {
                alert("Cannot give demonstrations for the goal state.");
            } else if (selectionOption === "active" && !firstSelection && index !== requestedState) {
                alert("Must give demonstration in the requested state.")
            } else {
                var action = null;
                if (envOption === "gridworld") {
                    action = prompt("Choose an action (U, D, L, R)");
                } else if (envOption === "driving") {
                    action = prompt("Choose an action (S, L, R)");
                }
                if (action !== null && action !== '') {
                    const actionDiv = document.createElement('div');
                    actionDiv.classList.add('action');
                    firstSelection = false;
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
                                requestedState = data.requested_state;
                                funFact = data.fun_fact;
                                displayDemoRequest(data.requested_state);
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

function displayDemoRequest(stateIdx) {
    if (stateIdx !== null) {
        statusUpdate.innerHTML = `<p>The agent would like another demonstration <strong>for state ${stateIdx}</strong>.</p>`;
    } else {
        statusUpdate.innerHTML = `<p>The agent would like another demonstration.</p>`;
    }
}

function displayFailedMessage() {
    statusUpdate.innerHTML = `<p>Unfortunately, the agent was not able to determine demonstration sufficiency. This could be the agent's fault, but still be sure to keep your demonstrations as consistent as possible with one another and with the reward function.</p><p>Thank you for playing! Click End Simulation to end this session and start another round.</p>`
}

function displaySufficiencyMessage(policy, goal_state) {
    statusUpdate.innerHTML = `
        <p>The agent has determined demonstration sufficiency! Here is the policy it learned.</p>
        <p>On a scale of 1 to 10, how well did the agent's learned policy match your intended policy or reward function? <strong>Please answer this, otherwise your experiment result will not be saved.</strong></p>
        <select id="user-evaluation">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9">9</option>
            <option value="10">10</option>
        </select>
        <button id="submit-evaluation">Submit Evaluation</button>
        <p></p>
    `;
    // Display arrows based on policy
    const squares = document.querySelectorAll('.square');
    squares.forEach((square, index) => {
        const actionDiv = square.querySelector('.action');
        const envOption = document.getElementById('environment-option').value;
        if (actionDiv) {
            square.removeChild(actionDiv);
        }
        if (index !== goal_state) {
            const arrowDiv = document.createElement('div');
            arrowDiv.classList.add('action');
            arrowDiv.style.color = 'white';
            arrowDiv.style.backgroundColor = 'transparent';
            const action = policy[index];
            switch (action) {
                case 0:
                    arrowDiv.textContent = '\u2191';
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
                default:
                    // Handle invalid action if needed
            }
            square.appendChild(arrowDiv);
        }
    });
    // Disable giving more demonstrations
    disableDemonstrations();
    // Add event listener for user's response submission
    const submitEvaluationButton = document.getElementById('submit-evaluation');
    submitEvaluationButton.addEventListener('click', () => {
        const userResponse = document.getElementById('user-evaluation').value;
        
        // Send the user's response to the server for storage
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
