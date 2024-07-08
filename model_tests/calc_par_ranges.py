"""
Utility to read in all model parameter ranges, 
calculate the 5th, 25th, 50th, and 75th percentiles of the parameter ranges, 
and export the results to an Excel file.
"""

import os
import numpy as np
import pandas as pd
import pymarrmot.models.models as models

# 1. Get a list of all the models
print("Getting a list of all the models.")
directory = 'C:/Users/ssheeder/Repos/pymarrmot/src/pymarrmot/models/models'
# Loop over all files in the directory
model_list = []
for filename in os.listdir(directory):
    # Check if the file is a Python file
    if filename.endswith('.py'):
        # Check if the file is a model file
        if filename.startswith('m_'):
            filename_no_ext = os.path.splitext(filename)[0]
            model_list.append(filename_no_ext)   

# 2. Initialize each model and get the parameter ranges
print("Initializing each model and getting the parameter ranges.")
par_ranges_all_models = []
for model in model_list:
    #print(f"Now starting model {model}.")
    m = getattr(models, model)()
    model_par_ranges = m.par_ranges
    par_ranges_all_models.append(model_par_ranges)

# 3. For each model, get the 5th, 25th, 50th, and 75th percentiles of the parameter ranges
print("Calculating the 5th, 25th, 50th, and 75th percentiles of the parameter ranges.")
par_ranges_5th_percentile = []
par_ranges_25th_percentile = []
par_ranges_50th_percentile = []
par_ranges_75th_percentile = []
for ranges in par_ranges_all_models:
    results_5th = []
    results_25th = []
    results_50th = []
    results_75th = []

    _type = type(ranges)
    if _type == np.ndarray:
        array_size = ranges.size
        array_size = int(array_size/2)
        # for each parameter
        for i in range(array_size):
            results_5th.append((ranges[i,1] - ranges[i,0])*0.05 + ranges[0,0])
            results_25th.append((ranges[i,1] - ranges[i,0])*0.25 + ranges[0,0])
            results_50th.append((ranges[i,1] - ranges[i,0])*0.50 + ranges[0,0])
            results_75th.append((ranges[i,1] - ranges[i,0])*0.75 + ranges[0,0])
    elif _type == list:
        list_length = len(ranges)
        # for each parameter
        for i in range(list_length):
            results_5th.append(((ranges[i])[1] - (ranges[i])[0])*0.05 + (ranges[i])[0])
            results_25th.append(((ranges[i])[1] - (ranges[i])[0])*0.25 + (ranges[i])[0])
            results_50th.append(((ranges[i])[1] - (ranges[i])[0])*0.50 + (ranges[i])[0])
            results_75th.append(((ranges[i])[1] - (ranges[i])[0])*0.75 + (ranges[i])[0])
    
    par_ranges_5th_percentile.append(results_5th)
    par_ranges_25th_percentile.append(results_25th)
    par_ranges_50th_percentile.append(results_50th)
    par_ranges_75th_percentile.append(results_75th)

# 4. export the results
print("Exporting the results to an Excel file.")
df = pd.DataFrame(par_ranges_5th_percentile)
df.to_csv("c:/temp/par_ranges_5th_percentile.csv", sep=',',index=False)
df = pd.DataFrame(par_ranges_25th_percentile)
df.to_csv("c:/temp/par_ranges_25th_percentile.csv", sep=',',index=False)
df = pd.DataFrame(par_ranges_50th_percentile)
df.to_csv("c:/temp/par_ranges_50th_percentile.csv", sep=',',index=False)
df = pd.DataFrame(par_ranges_75th_percentile)
df.to_csv("c:/temp/par_ranges_75th_percentile.csv", sep=',',index=False)

# 5. Notify
print("Calculations Complete!")