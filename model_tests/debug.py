def handle_warning(message, category, filename, lineno, file=None, line=None):
    print('A warning occurred:')
    print(message)
    print(filename + ', line ' + str(lineno))
    print('Do you wish to continue?')

    while True:
        response = input('y/n: ').lower()
        if response not in {'y', 'n'}:
            print('Not understood.')
        else:
            break

    if response == 'n':
        raise category(message)

import numpy as np
import pandas as pd
import pymarrmot.models.models as models

# Override the default warning handler
import warnings
warnings.showwarning = handle_warning

# User-defined options
climate_data_file = 'C:/Users/ssheeder/Repos/pymarrmot/model_tests/Forcing/c08194200.csv'
parameter_ranges_file = 'C:/Users/ssheeder/Repos/pymarrmot/model_tests/Parameters/par_ranges_5th_percentile.csv'
models_directory = 'C:/Users/ssheeder/Repos/pymarrmot/src/pymarrmot/models/models'

print("***************************************")
print("Preparing input data.")
print("***************************************")
# 1. Prepare the climate data
df = pd.read_csv(climate_data_file)
input_climatology = {
    'dates': df['datenum'].to_numpy(),      # Daily data: date in ??? format
    'precip': df['P [mm/d]'].to_numpy(),    # Daily data: P rate [mm/d]
    'temp': df['T [oC]'].to_numpy(),        # Daily data: mean T [degree C]
    'pet': df['PET [mm/d]'].to_numpy(),     # Daily data: Ep rate [mm/d]
}

# 2. Prepare the parameter values for each model
df_parameters = pd.read_csv(parameter_ranges_file)

# 3. The model we are debugging, and its position in the parameters file
model_list = []
model_list.append('m_43_gsmsocont_12p_6s')
i = 42 # m_XX minus 1

# 4. Define the solver nsettings
input_solver_opts = {
    'resnorm_tolerance': 0.1,
    'resnorm_maxiter': 6
}

# 5. Set up and run each model
results = []
results.append(['model name', 'parameter_values', 'output_ex', 'output_in', 'output_ss', 'output_wb'])

for model in model_list:
    print("***************************************")
    print(f"Now starting model {model}.")
    print("***************************************")
    m = getattr(models, model)()
    m.delta_t = 1
    model_range = m.par_ranges
    num_par = m.num_params
    num_store = m.num_stores
    input_theta = np.asarray(df_parameters.iloc[i])
    input_theta = input_theta[~np.isnan(input_theta)]
    input_s0 = np.ones(num_store)

    output_ex, output_in, output_ss, output_waterbalance = m.get_output(4,input_climatology, input_s0, input_theta, input_solver_opts)

    results.append([model, input_theta, output_ex, output_in, output_ss, output_waterbalance])
    i += 1

    # 6. Export the results
    results_df = pd.DataFrame()
    results_df['date'] = input_climatology['dates']
    i=0
    for result in results:
        if i != 0:
            name = result[0]
            ## output_ex
            for ex in result[2]:
                results_df[ex] = result[2][ex]
            for in_ in result[3]:
                results_df[in_] = result[3][in_]
            for ss in result[4]:
                results_df[ss] = result[4][ss] 
            results_df.to_csv(f'C:/Users/ssheeder/Repos/pymarrmot/model_tests/Results/results_{name}.csv', index=False)       
        i += 1

print("***************************************")
print("Model runs complete.")
print("***************************************")