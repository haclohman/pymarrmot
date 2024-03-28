import numpy as np
import matplotlib.pyplot as plt
#from MARRMoT_example_data import data_MARRMoT_examples
from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s
#from MARRMoT_model import feval
from pymarrmot.functions.objective_functions import of_kge, of_inverse_kge, of_mean_hilo_kge
import pandas as pd


# 1. Prepare data
# Load the data
# precipitation = data_MARRMoT_examples['precipitation']
# temperature = data_MARRMoT_examples['temperature']
# potential_evapotranspiration = data_MARRMoT_examples['potential_evapotranspiration']
df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet.csv')
# Create a climatology data input structure
input_climatology = {
    'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
    'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
    'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
}

# 2. Define the model settings
# model = 'm_29_hymod_5p_5s'

# Parameter values
input_theta = np.array([35,   # Soil moisture depth [mm]
                        3.7,  # Soil depth distribution parameter [-]
                        0.4,  # Fraction of soil moisture excess that goes to fast runoff [-]
                        0.25,  # Runoff coefficient of the upper three stores [d-1]
                        0.01])  # Runoff coefficient of the lower store [d-1]

# Initial storage values
input_s0 = np.array([15,  # Initial soil moisture storage [mm]
                     7,   # Initial fast flow 1 storage [mm]
                     3,   # Initial fast flow 2 storage [mm]
                     8,   # Initial fast flow 3 storage [mm]
                     22])  # Initial slow flow storage [mm]

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,  # Root-finding convergence tolerance
    'resnorm_maxiter': 6        # Maximum number of re-runs
}

# 4. Create a model object
# m = feval(model) - original code
m = m_29_hymod_5p_5s()

# Set up the model
m.delta_t = 1                        # time step size of the inputs: 1 [d]
m.theta = input_theta
m.input_climate = input_climatology
m.solver_opts = input_solver_opts
m.S0 = input_s0

# 5. Run the model and extract all outputs
(output_ex, output_in, output_ss, output_waterbalance) = m.get_output(m)

# 6. Analyze the outputs
# Prepare a time vector
#t = data_MARRMoT_examples['dates_as_datenum']
t = df['Date'].to_xarray

# Compare simulated and observed streamflow
tmp_obs = df['Q'].to_xarray
tmp_sim = output_ex['Q']
tmp_kge = of_kge(tmp_obs, tmp_sim)  # KGE on regular flows
tmp_kgei = of_inverse_kge(tmp_obs, tmp_sim)  # KGE on inverse flows
tmp_kgem = of_mean_hilo_kge(tmp_obs, tmp_sim)  # Average of KGE(Q) and KGE(1/Q)

# Plot simulated and observed streamflow
plt.figure(figsize=(10, 6))
plt.plot(t, tmp_obs, label='Observed', linewidth=2)
plt.plot(t, tmp_sim, label='Simulated', linewidth=2)
plt.legend()
plt.title('Kling-Gupta Efficiency = {:.2f}'.format(tmp_kge))
plt.ylabel('Streamflow [mm/d]')
plt.xlabel('Time [d]')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Investigate internal storage changes
plt.figure(figsize=(10, 10))
plt.subplot(311)
plt.plot(t, output_ss['S1'], label='Soil moisture [mm]')
plt.title('Simulated storages')
plt.ylabel('Soil moisture [mm]')
plt.xticks(rotation=45)

plt.subplot(312)
plt.plot(t, output_ss['S2'], label='Fast store 1')
plt.plot(t, output_ss['S3'], label='Fast store 2')
plt.plot(t, output_ss['S4'], label='Fast store 3')
plt.legend()
plt.ylabel('Fast stores [mm]')
plt.xticks(rotation=45)

plt.subplot(313)
plt.plot(t, output_ss['S5'], label='Slow store')
plt.ylabel('Slow store [mm]')
plt.xlabel('Time [d]')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
