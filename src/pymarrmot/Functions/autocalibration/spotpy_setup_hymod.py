from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

import pandas as pd
import numpy as np

class spotpy_setup(object):
    # Following example from https://spotpy.readthedocs.io/en/latest/How_to_link_a_model_to_SPOTPY/
    # Linking a model to SPOTPY is done by following five consecutive steps,
    # which are grouped in a spotpy_setup class

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'Smax': [1, 2000],   
        'b': [0.01, 10],        
        'a': [0.01, 1],         
        'kf': [0.01, 1],       
        'ks': [0.01, 1]        
        }
    
    # parameters that will be calibrated by Spotpy
    smax = Uniform(low=par_ranges['Smax'][0], high=par_ranges['Smax'][1], optguess=500)    # Smax, Maximum soil moisture storage [mm]
    b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=0.1725)          # b, Soil depth distribution parameter [-]
    a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0.8127)          # a, Runoff distribution fraction [-]
    kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.0404)       # kf, fast flow time parameter [d-1]
    ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5592)       # ks, base flow time parameter [d-1]   
    
    # class variables
    m = m_29_hymod_5p_5s()
    m.delta_t = 1/24  # time step in days (1 hour)
    trueObs = None

    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
        #initial storage values - set as average of the range
        input_s0 = []
        #loop over par_ranges dictionary and get average value for each key
        for key in self.par_ranges:
            input_s0.append((self.par_ranges[key][0]+self.par_ranges[key][1])/2) 

        #Define the solver settings
        input_solver_opts = {
            'resnorm_tolerance': 0.1,
            'resnorm_maxiter': 6
        }

        #Create a model object
        #m = m_29_hymod_5p_5s()

        #Time step of the model (in days)

        self.m.S0 = np.array(input_s0)
        self.m.solver_opts = input_solver_opts
        

        #USGS 03463300 - SOUTH TOE RIVER NEAR CELO, NC - usgsbasin-03463300_combined.parquet
        #USGS 02138500 - LINVILLE RIVER NEAR NEBO, NC - usgsbasin-02138500_combined.parquet
        #USGS 03441000 - DAVIDSON RIVER NEAR BREVARD, NC - usgsbasin-03441000_combined.parquet
        #USGS 03479000 - WATAUGA RIVER NEAR SUGAR GROVE, NC - usgsbasin-03479000_combined.parquet
        #USGS 03456100 - WEST FORK PIGEON RIVER AT BETHEL, NC - usgsbasin-03456100_combined.parquet

        df = pd.read_parquet('C:/Users/ssheeder/Repos/pymarrmot/forcing/pymarrmot/combined_forcing/usgsbasin-03463300_combined.parquet')

        # Create a climatology data input structure
        input_climatology = {
            'dates': df['value_time'].to_numpy(),  
            'precip': df['precip_mm'].to_numpy(),     
            'temp': df['temp_c'].to_numpy(),       
            'pet': df['pet_mm'].to_numpy()      
        }
        self.trueObs = df['discharge_mm'].tolist()
        #self.trueObs = df[['value_time','discharge_mm']]

        # # Clean up the evaluation and simulation data - remove any NaNs and missing values from evaluation dataset, and corresponding values from simulation dataset
        # nan_indices = np.argwhere(np.isnan(trueObs))
        # #missing_indices = np.argwhere(trueObs == "")
        # #unique_values = list(set(nan_indices + missing_indices))
        # if len(nan_indices) > 0:
        #     print(f"WARNING: {len(nan_indices)} NaN and missing values found in evaluation dataset. These values will be removed from the evaluation and simulation datasets before calculating model fitness.")
        #     trueObs = np.delete(trueObs, nan_indices)

        self.m.input_climate = input_climatology
            
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actually started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        return output_ex['Q'].tolist()
        # sim_q = output_ex['Q']
        

        # result_df = self.trueObs.copy()
        # result_df['simulated_discharge_mm'] = sim_q
        # result_df.drop(columns=['discharge_mm'], inplace=True)
        # return result_df
        #return sim_q.tolist()
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):
    #SPOTPY expects to get one or multiple values back, 
    #that define the performance of the model run
       #add simulation and evaluation lists to dataframe
        simulation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'simulated_discharge_mm': simulation})
        evaluation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'discharge_mm': evaluation})
       #left join on 'value_time' to align simulation and evaluation data
        merged_df = pd.merge(evaluation_df, simulation_df, on='value_time', how='left')
        evaluation_array = merged_df['discharge_mm'].to_numpy()
        simulation_array = merged_df['simulated_discharge_mm'].to_numpy()

        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation_array, simulation_array)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation_array, simulation_array)
            eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation]
            sim_inverse = [1000 if (i<=0) else 1/i for i in simulation]
            score2 = self.obj_func(eval_inverse, sim_inverse)
            result = (score + score2)/2
            like = -1*result
        return like