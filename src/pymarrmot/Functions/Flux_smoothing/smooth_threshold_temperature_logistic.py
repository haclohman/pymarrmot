import numpy as np

def smooth_threshold_temperature_logistic(P, T, Tt, r=0.01):
    """
    Logistic smoother for temperature threshold functions.

    Smooths the transition of threshold functions of the form:
    Snowfall = { P, if T <  Tt
                { 0, if T >= Tt

    By transforming the equation above to Sf = f(P, T, Tt, r):
    Sf = P * 1 / (1 + exp((T - Tt) / r))

    Inputs:
    P       : current precipitation
    T       : current temperature
    Tt      : threshold temperature below which snowfall occurs
    r       : smoothing parameter rho, default = 0.01

    NOTE: this function only outputs the multiplier. This needs to be
    applied to the proper flux outside of this function.
    """
    # Calculate multiplier
    out = 1 / (1 + np.exp((T - Tt) / r))

    return out