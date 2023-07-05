import time
import numpy as np
import pandas as pd
import datetime

def syed_schapire_bound(num_demos, gamma, num_features, delta):
    return 3/(1 - gamma) * np.sqrt(2/num_demos * np.log(2*num_features / delta))

def abbeel_bound(num_demos, gamma, num_features, delta):
    return 1/(1 - gamma) * np.sqrt(2*num_features / num_demos * np.log(2*num_features / delta))

delta = 0.05
gamma = 0.95
num_features = 4

start = time.time()
m = 1
epsilons = [0.05, 0.04, 0.03, 0.02, 0.01]
bounds_dict = {"bound": epsilons,
               "syed_schapire_num_demos": [], "syed_schapire_bound": [],
               "abbeel_num_demos": [], "abbeel_bound": []}

best_ss = syed_schapire_bound(m, gamma, num_features, delta)
best_abbeel = abbeel_bound(m, gamma, num_features, delta)
ss_eps_idx = 0
abbeel_eps_idx = 0
eval_ss = True
eval_abbeel = True
while eval_ss or eval_abbeel:
    m += 1
    if eval_ss:
        target_ss_eps = epsilons[ss_eps_idx]
        best_ss = syed_schapire_bound(m, gamma, num_features, delta)
        if best_ss <= target_ss_eps:
            bounds_dict["syed_schapire_num_demos"].append(m)
            bounds_dict["syed_schapire_bound"].append(best_ss)
            ss_eps_idx += 1
            if ss_eps_idx == len(epsilons):
                eval_ss = False
    if eval_abbeel:
        target_abbeel_eps = epsilons[abbeel_eps_idx]
        best_abbeel = abbeel_bound(m, gamma, num_features, delta)
        if best_abbeel <= target_abbeel_eps:
            bounds_dict["abbeel_num_demos"].append(m)
            bounds_dict["abbeel_bound"].append(best_abbeel)
            abbeel_eps_idx += 1
            if abbeel_eps_idx == len(epsilons):
                eval_abbeel = False
end = time.time()

bounds_demo_df = pd.DataFrame(bounds_dict)
bounds_demo_df.to_csv("bounds_demo_df.csv", index = False)
print("It took", str(datetime.timedelta(seconds = end - start)), "to run", m, "demos.")