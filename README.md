# Demonstration Sufficiency User Study

Thank you for participating in the user study for our demonstration sufficiency experiments!

## Setup
This user study will require a little bit of installation on your end before beginning:
1. Clone the Github repo:
```
git clone https://github.com/dsbrown1331/birl.git
```
2. Retrieve the branches.
```
git fetch
```
3. Switch to the branch with the user study code.
```
git checkout userstudy
```
4. Get the latest updates.
```
git pull
```
6. Make a new branch for your experiments. **Please stay on this branch when running the experiment**. Result files are written to the `results/` folder after each experiment, and this ensures everyone's files do not override one another's. It also makes it easier for us to access the results by just checking out your branch! Please make sure the branch follows the naming format, e.g. `results-atrinh` or `results-elephant`. 
```
git checkout -b results-<any unique identifier you want>
```
7. Push this branch to the repo.
```
git push --set-upstream origin <your branch name>
```

## How to Run
The user study will be conducted via a Flask web app. We understand that not everyone will have the necessary libraries to run this, so we recommend running it in a conda environment so that any additional libraries can be installed cleanly in there and not affect other existing packages. Here's how to set it up:
1. Install `conda`, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). This install will modify the `PATH` variable in your bashrc or bashprofile. **You need to open a new terminal for that path change to take place (to be able to find `conda` in the next step).**
2. Create a conda environment that will contain python3:
```
conda create -n demo_sufficiency python=3.11
```
3. Activate the environment. **Do this every time you open a new terminal and want to run the user study**:
```
conda activate demo_sufficiency
```
4. Install the requirements into this environment:
```
pip install -r requirements.txt
```
5. To run the user study, go into the directory where the web app lives:
```
cd user_interface
```
6. Run the app. It should start up a development server, which you can access by opening a browser tab and going to `localhost:5000`.
```
python3 app.py
```
7. Have fun! All instructions should be on the user study interface itself. Please read and follow them carefully. The app has been tested extensively, but if at any point there is a crash, just do Ctrl+C on the terminal to end the session then restart it again.

## Submitting Results
You will notice that there is a `results/` directory inside `user_interface`. This is where all experiment results will be written to once they are completed. We suggest that you push these experiments to your branch every now and then, so that we can analyze them as they come. Make sure you're in the `user_interface/` directory, then do the following:
1. First, check to see if you've made any edits to the code. **Undo them if so.** The only changes that should show up when you run this command are the result files.
```
git status
```
2. Stage, commit, and push your results.
```
git add .
git commit -m <any message here>
git push
```

Thank you very much for taking your time and participating! If you have any questions, please ask Tu (Alina) Trinh via Slack or [email](mailto:tutrinh@berkeley.edu).
