
import numpy as np

def uh_4_full(d_base, delta_t):
    """
    Unit hydrograph [days] with a triangle (linear).
    
    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter.
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.
    
    Parameters:
    d_base : float
        Time base of routing delay [d].
    delta_t : numpy.ndarray
        Time step size [d].
    
    Returns:
    UH : numpy.ndarray
        Unit hydrograph [nx2].
        uh's first row contains coefficients to split flow at each
        of n timesteps forward, the second row contains zeros now,
        these are the still-to-flow values.
    
    Unit hydrograph spreads the input volume over a time period delay.
    Percentage runoff goes up, peaks, and goes down again.
    Example: d_base = 3.8 [days], delta_t = 1.
        UH(1) = 0.14  [% of inflow]
        UH(2) = 0.41
        UH(3) = 0.36
        UH(4) = 0.09
    """
    
    UH = np.zeros((len(delta_t), 2))
    T = d_base / 2
    N = len(delta_t)
    for i in range(N):
        if delta_t[i] < T:
            UH[i, 0] = 2 * delta_t[i] / d_base
        else:
            UH[i, 0] = 2 - 2 * delta_t[i] / d_base
    
    return UH
