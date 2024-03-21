
import numpy as np

def saturation_13(p1, p2, S, In):
    # Saturation excess flow from a store with different degrees of saturation (normal distribution variant)
    # Inputs:
    #   p1 - soil depth where 50% of catchment contributes to overland flow [mm]
    #   p2 - soil depth where 16% of catchment contributes to overland flow [mm]
    #   S - current storage [mm]
    #   In - incoming flux [mm/d]
    
    out = In * norm.cdf(np.log10(max(0, S) / p1) / np.log10(p1 / p2))
    return out
