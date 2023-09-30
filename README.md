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
```
2. On the off chance that we needed to make edits to the `userstudy` branch, we will notify you. To get the latest updates in that case, just pull.
```
git pull
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
5. Go into the `user_interface` folder and run the app. It should start up a development server, which you can access by opening a browser tab and going to `localhost:5000`. If the commands below introduce import errors, append the path to the `birl` directory to your `sys.path`, for example, `sys.path.append("/Users/tutrinh/birl/")`. You can do this inside `app.py`, under the `import sys` statement.
```
cd user_interface
python3 app.py
```
6. Have fun! All instructions should be on the user study interface itself. Please read and follow them carefully. The app has been tested, but if at any point there is a crash, just end the server with Ctrl+C then restart it. You can start and stop the server as much as you want, **as long as it's between complete simulations**—otherwise your work will be lost!
