
import numpy as np

def uh_7_uniform(d_base, delta_t):
    """
    Unit Hydrograph with Uniform Spread
    
    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.
    
    Inputs:
    - d_base: time base of routing delay [days]
    - delta_t: time step size [days]
    
    Output:
    - UH: unit hydrograph [nx2]
          uh's first row contains coefficients to split flow at each
          of n timesteps forward, the second row contains zeros now,
          these are the still-to-flow values.
          
    The unit hydrograph spreads the input volume over a time period delay.
    """
    # Calculate delay based on time base and time step size
    delay = d_base / delta_t
    tt = np.arange(1, np.ceil(delay) + 1)
    
    # Perform further calculations according to the MATLAB function
    
    # Placeholder return values
    UH = np.array([[0.0], [0.0]])

    return UH
