import time
import numpy as np
import pandas as pd
import datetime

bounds_dict = {"num_demos": [], "syed_schapire": [], "abbeel": []}

def syed_schapire_bound(num_demos, gamma, num_features, delta):
    return 3/(1 - gamma) * np.sqrt(2/num_demos * np.log(2*num_features / delta))

def abbeel_bound(num_demos, gamma, num_features, delta):
    return 1/(1 - gamma) * np.sqrt(2*num_features / num_demos * np.log(2*num_features / delta))

delta = 0.05
gamma = 0.95
num_features = 4

start = time.time()
m = 1
target_eps = 0.01
best_ss = syed_schapire_bound(m, gamma, num_features, delta)
best_abbeel = abbeel_bound(m, gamma, num_features, delta)
bounds_dict["num_demos"].append(m)
bounds_dict["syed_schapire"].append(best_ss)
bounds_dict["abbeel"].append(best_abbeel)
while best_ss > target_eps or best_abbeel > target_eps:
  m += 1
  bounds_dict["num_demos"].append(m)
  best_ss = syed_schapire_bound(m, gamma, num_features, delta)
  bounds_dict["syed_schapire"].append(best_ss)
  best_abbeel = abbeel_bound(m, gamma, num_features, delta)
  bounds_dict["abbeel"].append(best_abbeel)
end = time.time()

bounds_df = pd.DataFrame(bounds_dict)
bounds_df.to_csv("bounds_df.csv")
print("It took", str(datetime.timedelta(seconds = end - start)), "to run", m, "demos.")
