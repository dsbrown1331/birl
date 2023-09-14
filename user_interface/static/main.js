const startForm = document.getElementById("start-form"); // Get the form element
const gridContainer = document.getElementById('grid-container');
const endButton = document.getElementById('end-button'); // Get the end button

startForm.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent the form submission
    fetch('/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            'teaching_option': document.getElementById('teaching-option').value,
            'selection_option': document.getElementById('selection-option').value
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
        if (data.reward_function !== null && data.reward_function !== undefined) {
            const rewardFunctionString = data.reward_function.map((item, index) => {
                let colorClass = '';
                let colorLabel = '';
                switch (index + 1) {
                    case 1:
                        colorClass = 'red-text';
                        colorLabel = 'Red';
                        break;
                    case 2:
                        colorClass = 'blue-text';
                        colorLabel = 'Blue';
                        break;
                    case 3:
                        colorClass = 'green-text';
                        colorLabel = 'Green';
                        break;
                    case 4:
                        colorClass = 'purple-text';
                        colorLabel = 'Purple';
                        break;
                    case 5:
                        colorClass = 'black-text';
                        colorLabel = 'Goal';
                        break;
                    default:
                        break;
                }
                return `<span class="${colorClass}">${colorLabel}: ${item}</span>`;
            }).join(',<br>');
    
            // Update the <span> element with the class
            document.getElementById("reward-function-vector").innerHTML = "Here is the reward function you should follow:<br>" + "[" + rewardFunctionString + "]" + "<br>Reminder that on your way to the goal, you should be avoiding states with lower reward values and opting to go through states with higher reward values instead.";
        }
    })
    .catch(function (error) {
        console.error("There was an error with /start:", error);
    });
});

startForm.addEventListener('submit', (event) => {
    event.preventDefault(); // Prevent the form submission
    // Toggle the visibility of the grid container
    if (gridContainer.style.display === 'none' || gridContainer.style.display === '') {
        gridContainer.style.display = 'block';
        // Show the undo and end buttons
        endButton.style.display = 'block';
    } else {
        gridContainer.style.display = 'none';
        // Hide the undo and end buttons
        endButton.style.display = 'none';
    }
});

endButton.addEventListener('click', () => {
    // Send a request to end the simulation and reset data
    fetch('/end_simulation', {
        method: 'POST',
    })
    .then(response => response.text())
    .then(message => {
        // Handle the response message as needed
        console.log(message);
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
    });
    gridContainer.style.display = 'none';
    // Hide the undo and end buttons
    endButton.style.display = 'none';
});
    
const squares = document.querySelectorAll('.square');
squares.forEach((square, index) => {
    square.addEventListener('click', () => {
        if (index === 23) {
            alert("Cannot give demonstrations for the terminal goal state.");
        } else {
            const action = prompt('Choose an action (U, D, L, R):');
            if (action !== null && action !== '') {
                const actionDiv = document.createElement('div');
                actionDiv.classList.add('action');
                
                // Set arrow symbols based on the user-selected action
                switch (action.toUpperCase()) {
                    case 'U':
                        actionDiv.textContent = '\u2191'; // Up arrow
                        break;
                    case 'D':
                        actionDiv.textContent = '\u2193'; // Down arrow
                        break;
                    case 'L':
                        actionDiv.textContent = '\u2190'; // Left arrow
                        break;
                    case 'R':
                        actionDiv.textContent = '\u2192'; // Right arrow
                        break;
                    default:
                        actionDiv.textContent = action;
                }
                
                square.appendChild(actionDiv);
                
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
                .then(response => response.text())
                .then(message => console.log(message));
            }
        }
    });
});