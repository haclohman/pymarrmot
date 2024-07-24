import os
import pandas as pd

results_directory = 'C:/Users/ssheeder/Repos/pymarrmot/model_tests/Results/p05/'
matlab_directory = results_directory + 'matlab'
python_directory = results_directory + 'python'

df_output = pd.DataFrame()

i=0
for filename in os.listdir(results_directory + '/matlab'):
    if filename.endswith('.csv'):
        
        print(f"Processing {filename}")
        df_matlab = pd.read_csv(f'{matlab_directory}/{filename}')
        df_python = pd.read_csv(f'{python_directory}/{filename}')
        
        if i==0:
            df_output['date'] = df_matlab['date']
            i=1
        
        model = filename.rsplit(".", 1)[0]
        df_output[f'{model}-matlab-Q'] = df_matlab['ex_Q']
        df_output[f'{model}-python-Q'] = df_python['ex_Q']

df_output.to_csv(f'{results_directory}Summary.csv', index=False)

print('Processing complete')