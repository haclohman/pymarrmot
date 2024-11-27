from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

import os
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
    b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=0.1725)     # b, Soil depth distribution parameter [-]
    a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0.8127)      # a, Runoff distribution fraction [-]
    kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.0404)     # kf, fast flow time parameter [d-1]
    ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5592)     # ks, base flow time parameter [d-1]   
    
    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
        #initial storage values - set as average of the range
    input_s0 = []
    #loop over par_ranges dictionary and get average value for each key
    for key in par_ranges:
        input_s0.append((par_ranges[key][0]+par_ranges[key][1])/2) 

    #Define the solver settings
    input_solver_opts = {
        'resnorm_tolerance': 0.1,
        'resnorm_maxiter': 6
    }

    #Create a model object
    m = m_29_hymod_5p_5s()
    m.delta_t = 1

    df = pd.read_csv('C:/Users/ssheeder/Repos/pymarrmot/examples/Example_DataSet_NewDateFormat.csv')
    # Create a climatology data input structure
    input_climatology = {
        'dates': df['date'].to_numpy(),      # Daily data: date in 'm/d/yyyy' format
        'precip': df['precip'].to_numpy(),   # Daily data: P rate [mm/d]
        'temp': df['temp'].to_numpy(),       # Daily data: mean T [degree C]
        'pet': df['pet'].to_numpy(),         # Daily data: Ep rate [mm/d]
    }
    trueObs = df['Q'].to_list()
    m.input_climate = input_climatology
    
    m.solver_opts = input_solver_opts
    m.S0 = np.array(input_s0)
        
  
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        return output_ex['Q'].tolist()
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):
    #SPOTPY expects to get one or multiple values back, 
    #that define the performance of the model run
        
        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation, simulation)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation, simulation)
            eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation]
            sim_inverse = [1000 if (i<=0) else 1/i for i in simulation]
            score2 = self.obj_func(eval_inverse, sim_inverse)
            result = (score + score2)/2
            like = -1*result

            # Calculation of kge 
            # result = self.obj_func(evaluation, simulation)
            # like = -1*result


        return like