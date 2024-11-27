"""
This file is part of the Python Modular Assessment of Rainfall-Runoff Models
Toolbox (pyMARRMoT).
pyMARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
WARRANTY. See <https://www.gnu.org/licenses/> for details.

Contact: ssheeder@rti.org

This example workflow contains an example application of autocalibration of a
single model to a single catchment. The model utilized in this example is the
Hymod model, which is a conceptual rainfall-runoff model with five stores and
five fluxes. The model is calibrated to a single catchment using the SCE-UA
optimization algorithm implemented in the SPOTPY package. 

This example file calls the 'spotpy_setup' class, which is a wrapper around
the Hymod model and the SPOTPY package. The 'spotpy_setup' class contains
methods to set up the model, define the parameter ranges, and define the
objective function. The 'spotpy_setup' class is used to set up the model for
calibration and to run the SCE-UA optimization algorithm.
"""

import pandas as pd
import spotpy
import spotpy.objectivefunctions as of
from pymarrmot.functions.autocalibration.spotpy_setup_II import spotpy_setup as setup
import matplotlib.pyplot as plt

# 1. Prepare data
# df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet_NewDateFormat.csv')

# Create a climatology data input structure
# input_climatology = {
#     'dates': df['Date'].to_numpy(),      # Daily data: date in ??? format
#     'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
#     'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
#     'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
# }

# 2. Define the model settings
model_list = ['m_29_hymod_5p_5s',
              'm_01_collie1_1p_1s',
              'm_27_tank_12p_4s']

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,
    'resnorm_maxiter': 6
}

for model in model_list:
    print(f"Now starting model {model}.")

    #Define the solver settings
    input_solver_opts = {
        'resnorm_tolerance': 0.1,
        'resnorm_maxiter': 6
    }

    #input data
    df = pd.read_csv('C:/Users/ssheeder/Repos/pymarrmot/examples/Example_DataSet_NewDateFormat.csv')
    # Create a climatology data input structure
    input_climatology = {
        'dates': df['date'].to_numpy(),      # Daily data: date in 'm/d/yyyy' format
        'precip': df['precip'].to_numpy(),   # Daily data: P rate [mm/d]
        'temp': df['temp'].to_numpy(),       # Daily data: mean T [degree C]
        'pet': df['pet'].to_numpy(),         # Daily data: Ep rate [mm/d]
    }
    trueObs = df['Q'].to_list()

    #Set the number of model runs
    reps = 5000
    spotpy_setup = setup(of.kge_non_parametric, model, input_climatology, trueObs, input_solver_opts)
    sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA_hymod', dbformat='csv')
    sampler.sample(reps, ngs=7, kstop=3, peps=0.1, pcento=0.1)
    results = spotpy.analyser.load_csv_results('SCEUA_hymod')

    #find the run_id with the minimal objective function value
    bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)

    #Then we select the best model run:
    best_model_run = results[bestindex]

    #And filter results for simulation results only:
    fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])

    #Timeseries plot of the best model run
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
    ax.plot(spotpy_setup.evaluation(),color='red',markersize=3, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel ('Discharge [cfs]')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_hymod_best.png',dpi=300)

    #FDC plot of the best model run, exceedance probability on the x-axis and discharge on the y-axis
    #Sort the simulated and observed data, and calculate the exceedance probability
    sim_sorted = best_simulation
    sim_sorted.sort(reverse=True)
    obs_sorted = spotpy_setup.evaluation()
    obs_sorted.sort(reverse=True)
    exceedance_prob = [(x/(len(sim_sorted) + 1)*100) for x in range(len(sim_sorted))]
    #get the FDC plot
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    ax.plot(exceedance_prob, sim_sorted, color='black', linestyle='solid', label='Simulated')
    ax.plot(exceedance_prob, obs_sorted, color='red', linestyle='solid', linewidth= 2,label='Observed')
    plt.yscale('log')
    plt.xlabel('Exceedance Probability')
    plt.ylabel('Discharge [cfs]')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_hymod_best_FDC.png', dpi=300)