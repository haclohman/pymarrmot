
import numpy as np

def uh_2_full(d_base, delta_t):
    """
    Calculate a unit hydrograph with a full bell curve.
    
    Parameters
    ----------
    d_base : float
        Time base of routing delay [days].
    delta_t : float
        Time step size [days].
    
    Returns
    -------
    np.ndarray
        Unit hydrograph.
        uh's first row contains coefficients to split flow at each
        of n timesteps forward, the second row contains zeros now,
        these are the still-to-flow values.
    """
    # Code converted from MATLAB to Python
    x = np.linspace(delta_t, 4 * d_base, int(4 * d_base / delta_t))
    xx = (2 * x) / d_base
    UH = 0.5 * np.exp(-xx) - np.exp(-2*xx) + 0.5 * np.exp(-2 * x / d_base)
    UH /= np.sum(UH)
    
    return np.vstack(UH, np.zeros_like(UH))

# Testing the function with example values
d_base_example = 3.8
delta_t_example = 1
unit_hydrograph = uh_2_full(d_base_example, delta_t_example)
