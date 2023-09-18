# Demonstration Sufficiency User Study

Thank you for participating in the user study for our demonstration sufficiency experiments!

## Setup
This user study will require a little bit of installation on your end before beginning:
1. Clone the Github repo and get the user study code.
```
git clone https://github.com/dsbrown1331/birl.git
cd birl
git fetch
git checkout userstudy
git pull
```
2. Make a new branch for your experiments. **Please stay on this branch when running the experiment**. Result files are written to the `results/` folder after each experiment, so it's easier for us to access the results by just checking out your branch! Please make sure that whatever name you give your branch, it should have "results-" prepended, e.g. `results-atrinh` or `results-narwhal`. 
```
git checkout -b results-<any unique identifier you want>
git push --set-upstream origin <your branch name>
```

## How to Run
The user study will be conducted via a Flask web app. We understand that not everyone will have the necessary libraries to run this, so we recommend installing and running in a conda environment. Here's how to set it up:
1. Install `conda`, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). This install will modify the `PATH` variable in your bashrc or bashprofile. **You need to open a new terminal for that path change to take place (to be able to find `conda` in the next step).**
2. Create a conda environment that will contain python3.
```
conda create -n demo_sufficiency python=3.11
```
3. Activate the environment. **Do this every time you open a new terminal and want to run the user study**.
```
conda activate demo_sufficiency
```
4. **Make sure you are inside the `birl` directory.** Install the requirements into this environment.
```
pip install -r requirements.txt
```
5. **Make sure you are on your branch that you made above.** Run the app. It should start up a development server, which you can access by opening a browser tab and going to `localhost:5000`.
```
cd user_interface
python3 app.py
```
If the above introduces import errors, append the path to the `birl` directory to your `sys.path`, for example, `/Users/tutrinh/birl/`. You can do this inside `app.py` if you want, but make sure to delete these lines before you push your results.
6. Have fun! All instructions should be on the user study interface itself. Please read and follow them carefully. The app has been tested, but if at any point there is a crash, just end the server with Ctrl+C then restart it. **At the end of each simulation, you will find a question asking for your evaluation of the agent. Results will only be recorded if you submit this evaluation, so please do so!**

## Submitting Results
All experiment results are written to `user_interface/results/`. Please push these results to your branch as often as you can, so that we can analyze them as they come. To do so, make sure you're in the `user_interface/` directory, then do the following:
1. Check to see what changes are in the codebase. The only changes that should show up are the new result files. **If you've made any other edits, including the `sys.path` modification, undo them with `git checkout -- <file>`.** 
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
