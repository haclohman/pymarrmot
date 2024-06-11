"""
This is a test to run models 01 to 10 with the same input data and solver options.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymarrmot.models.models.m_01_collie1_1p_1s import m_01_collie1_1p_1s
from pymarrmot.models.models.m_02_wetland_4p_1s import m_02_wetland_4p_1s
from pymarrmot.models.models.m_03_collie2_4p_1s import m_03_collie2_4p_1s
from pymarrmot.models.models.m_04_newzealand1_6p_1s import m_04_newzealand1_6p_1s
from pymarrmot.models.models.m_05_ihacres_7p_1s import m_05_ihacres_7p_1s
from pymarrmot.models.models.m_06_alpine1_4p_2s import m_06_alpine1_4p_2s
from pymarrmot.models.models.m_07_gr4j_4p_2s import m_07_gr4j_4p_2s
from pymarrmot.models.models.m_08_us1_5p_2s import m_08_us1_5p_2s
from pymarrmot.models.models.m_09_susannah1_6p_2s import m_09_susannah1_6p_2s
from pymarrmot.models.models.m_10_susannah2_6p_2s import m_10_susannah2_6p_2s

# 1. Prepare data
df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet.csv')

# Create a climatology data input structure
input_climatology = {
    'dates': df['Date'].to_numpy(),      # Daily data: date in ??? format
    'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
    'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
    'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
}

# 2. Define the model settings
model_list =   [m_01_collie1_1p_1s,
                m_02_wetland_4p_1s,
                m_03_collie2_4p_1s,
                m_04_newzealand1_6p_1s,
                m_05_ihacres_7p_1s,
                m_06_alpine1_4p_2s,
                m_07_gr4j_4p_2s,
                m_08_us1_5p_2s,
                m_09_susannah1_6p_2s,
                m_10_susannah2_6p_2s]

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,
    'resnorm_maxiter': 6
}

# Create and run all model objects
results_sampling = []

results_sampling.append(['model name', 'parameter_values', 'output_ex', 'output_in', 'output_ss', 'output_wb'])

for model in model_list:
    print(f"Now starting model {model}.")
    m = model()
    m.delta_t = 1
    model_range = np.array(m.par_ranges)
    num_par = m.num_params
    num_store = m.num_stores

    input_theta = model_range[:, 0] + np.random.rand(num_par) * (model_range[:, 1] - model_range[:, 0])
    input_s0 = np.zeros(num_store)

    output_ex, output_in, output_ss, output_waterbalance = m.get_output(4,input_climatology, input_s0, input_theta, input_solver_opts)

    results_sampling.append([model, input_theta, output_ex, output_in, output_ss, output_waterbalance])

# 6. Analyze the outputs
t = input_climatology['dates']
streamflow_observed = df['Q'].to_numpy()

plt.figure(figsize=(10, 6))
plt.plot(t, streamflow_observed, 'k', linewidth=2)
for i in range(len(model_list)):
    plt.plot(t, results_sampling[i + 1][2]['Q'])

plt.legend(['Observed'] + model_list, loc='upper right')
plt.title('Model sampling results')
plt.ylabel('Streamflow [mm/d]')
plt.xlabel('Time [d]')
plt.grid(True)
plt.xticks(rotation=45)
plt.ylim([0, 70])
plt.tight_layout()
plt.show(block=True)

