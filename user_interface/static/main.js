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
    .then(response => console.log(response.text()))
    .then(message => console.log(message));
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
    // Implement logic to end the simulation and store logs
    // Redirect the user to the root page
    window.location.href = '/';
});
    
const squares = document.querySelectorAll('.square');
squares.forEach((square, index) => {
    square.addEventListener('click', () => {
        if (index === 23) {
            alert("Cannot choose actions for the terminal goal state.");
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