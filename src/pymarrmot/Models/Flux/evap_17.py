
def evap_17(p1, S, Ep):
    '''
    Flux function
    Description: Scaled evaporation from a store that allows negative values
    Constraints: -
    Inputs: 
        p1 - linear scaling parameter [mm-1]
        S - current storage [mm]
        Ep - potential evapotranspiration rate [mm/d]
    '''
    import numpy as np
    out = 1 / (1 + np.exp(-1 * p1 * S)) * Ep
    return out
