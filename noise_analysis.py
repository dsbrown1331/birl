import random
import mdp_utils
import mdp_worlds
import bayesian_irl
from mdp import FeatureMDP, DrivingSimulator
import copy
from scipy.stats import norm
import numpy as np
import math
import sys
import argparse
import pandas as pd
import os
import json
import time
import re

def get_user_studies_df():
    user_studies = pd.DataFrame()
    folder_path = "../ijcai-hri/user_study_results/"
    for filename in os.listdir(folder_path):
        if filename.startswith("success-") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                    # Check if the DataFrame has columns not present in the JSON
                    missing_cols = set(user_studies.columns) - set(data.keys())
                    if missing_cols:
                        for col in missing_cols:
                            data[col] = None  # Add missing columns with np.nan values
                    # Check if the JSON has columns not present in the DataFrame
                    extra_cols = set(data.keys()) - set(user_studies.columns)
                    if extra_cols:
                        for col in extra_cols:
                            user_studies[col] = None  # Add columns to the DataFrame with np.nan values
                    # Append the data to the master DataFrame
                    user_studies = user_studies.append(data, ignore_index = True)
                    user_studies.loc[user_studies.index[-1], "file"] = os.path.splitext(filename)[0]
                except Exception as e:
                    print(e)
                    print(file_path)
    return user_studies

def get_user_studies_partitions(user_studies):
    partitions = {}
    # Metrics: num demos, F1 score
    usable = user_studies[~(user_studies["pct_states"].isna())
                        & ~(user_studies["user_evaluation"].isna())
                        & (user_studies["demo_suff"] == True)]
    usable["pct_states"] = pd.to_numeric(usable["pct_states"])
    usable["user_evaluation"] = pd.to_numeric(usable["user_evaluation"])

    ours = usable[usable["methodology"] == "ours"]
    ours_gridworld = ours[ours["is_gridworld"] == True]
    ours_driving = ours[ours["is_gridworld"] == False]
    MAPs = usable[usable["methodology"] == "MAP"]
    MAPs_gridworld = MAPs[MAPs["is_gridworld"] == True]
    MAPs_driving = MAPs[MAPs["is_gridworld"] == False]
    held_outs = usable[usable["methodology"] == "held_out"]
    held_outs_gridworld = held_outs[held_outs["is_gridworld"] == True]
    held_outs_driving = held_outs[held_outs["is_gridworld"] == False]
    gridworlds = usable[usable["is_gridworld"] == True]
    drivings = usable[usable["is_gridworld"] == False]

    partitions["ours"] = ours
    partitions["ours_gridworld"] = ours_gridworld
    partitions["ours_driving"] = ours_driving
    partitions["MAPs"] = MAPs
    partitions["MAPs_gridworld"] = MAPs_gridworld
    partitions["MAPS_driving"] = MAPs_driving
    partitions["held_outs"] = held_outs
    partitions["held_outs_gridworld"] = held_outs_gridworld
    partitions["held_outs_driving"] = held_outs_driving
    partitions["gridworlds"] = gridworlds
    partitions["drivings"] = drivings
    return partitions


def retrieve_reward(df):
    num_rows = 0
    len_df = len(df)
    def retrieve_reward_row(row):
        nonlocal num_rows
        start = time.time()
        if row["is_gridworld"] == True:
            env_type = "gridworld"
            env = mdp_worlds.random_feature_mdp(5, 5, 4)
            # array([-0.5, -0.5, -0.5,  0.5])
            motorists = None
        else:
            env_type = "driving"
            env = mdp_worlds.random_driving_simulator(5, chosen_reward = np.array([0.4472136, 0.4472136, 0.4472136, 0.4472136, 0.4472136]))  # normalizing a [1, 1, 1, 1, 1] reward
            motorists = env.motorists
        true_policy_demos = list(enumerate(row["true_optimal_policy"]))
        birl = bayesian_irl.BIRL(env, true_policy_demos, 10)
        birl.run_mcmc(500, 0.5, adaptive = True)
        reward = np.asarray(birl.get_map_solution())
        state_features = env.state_features
        end = time.time()
        num_rows += 1
        print(f"{round(num_rows / len_df, 2)}% done, env was {env_type}, reward was {reward}, took {round((end - start) / 60, 2)} minutes")
        return pd.Series([reward, state_features, motorists])
    df[["reward", "state_features", "motorists"]] = df.apply(retrieve_reward_row, axis = 1)
    return df

def calculate_noise(df):
    num_rows = 0
    len_df = len(df)
    def calculate_noise_row(row):
        nonlocal num_rows
        start = time.time()
        reward = np.array([float(r) for r in re.findall(r"([0-9-.]+)", row["reward"])])
        if row["is_gridworld"] == True:
            env_type = "gridworld"
            state_features = eval(row["state_features"])
            env = FeatureMDP(5, 5, 4, [], reward, state_features, gamma = 0.95, noise = 0, driving = False)
        else:
            env_type = "driving"
            motorists = [int(r) for r in re.findall(r"([0-9-.]+)", row["motorists"])]
            env = DrivingSimulator(5, [], reward, motorists = motorists, police = [], gamma = 0.95, noise = 0)
        q_values = mdp_utils.calculate_q_values(env, epsilon = 0.0001)
        given_demos = eval(row["given_demos"])
        noisy_demos = 0
        for demo in given_demos:
            state, action = demo[0], demo[1]
            if action not in mdp_utils.arg_max_set(q_values[state], 0.0001):
                noisy_demos += 1
        end = time.time()
        num_rows += 1
        print(f"{round(num_rows / len_df, 2)}% done, env was {env_type}, {noisy_demos} noisy demos, took {round((end - start) / 60, 2)} minutes")
        return pd.Series([noisy_demos])
    df[["num_noisy_demos"]] = df.apply(calculate_noise_row, axis = 1)
    return df

def noise_finder(df, sub = 0):
    avg_noise = []
    gridworld_noise = []
    driving_noise = []
    for _, row in df.iterrows():
        noise = max(0, row['num_noisy_demos'] - sub) / row['num_demos']
        avg_noise.append(noise)
        if row["is_gridworld"] == True:
            gridworld_noise.append(noise)
        else:
            driving_noise.append(noise)
    print("Total:", np.mean(avg_noise))
    print("Gridworld:", np.mean(gridworld_noise))
    print("Driving:", np.mean(driving_noise))


if __name__ == "__main__":
    stage = int(sys.argv[1])
    if stage == 1:
        user_studies = get_user_studies_df()
        user_studies.to_csv("user_studies.csv", index = False)
        partitions = get_user_studies_partitions(user_studies)
        # partitions["ours"].to_csv("ours.csv", index = False)
        # ours = pd.read_csv("ours.csv")
        partitions["ours"] = retrieve_reward(partitions["ours"])
        assert "reward" in partitions["ours"].columns.values
        partitions["ours"].to_csv("ours_with_reward_and_state_features.csv", index = False)
    elif stage == 2:
        df = pd.read_csv("ours_with_complete_env.csv")
        df_with_noise = calculate_noise(df)
        df_with_noise.to_csv("ours_with_noise.csv", index = False)
    elif stage == 3:
        df = pd.read_csv("ours_with_noise.csv")
        noise_finder(df, 1.5)  # g: 2 (14%), d: 3 (8%)
    elif stage == 4:
        user_studies = pd.read_csv("user_studies.csv")
        partitions = get_user_studies_partitions(user_studies)
        df = partitions["ours"]
        def quick_calculate_noisy_demos(row):
            given_demos = eval(row["given_demos"])
            true_optimal_policy = list(enumerate(row["true_optimal_policy"]))
            noisy_demos = 0
            for demo in given_demos:
                state, action = demo[0], demo[1]
                if action != true_optimal_policy[state]:
                    noisy_demos += 1
            return noisy_demos
        df["num_noisy_demos"] = df.apply(quick_calculate_noisy_demos, axis = 1)
        df.to_csv("ours_with_quick_calc_noise.csv", index = False)
        noise_finder(df, 5)