
import numpy as np

def uh_3_half(d_base: float, delta_t: float) -> np.ndarray:
    """
    Calculate unit hydrograph [days] with half a triangle (linear).

    Parameters:
    d_base (float): time base of routing delay [d]
    delta_t (float): time step size [d]

    Returns:
    np.ndarray: Unit hydrograph [nx2]
               uh's first row contains coefficients to split flow at each
               of n timesteps forward, the second row contains zeros now,
               these are the still-to-flow values.
    """

    # Unit hydrograph spreads the input volume over a time period delay.
    # Percentage of input returned only increases.
    # Example: d_base = 3.8 [days], delta_t = 1
    # UH(1) = 0.04  [% of inflow]
    # UH(2) = 0.17
    # UH(3) = 0.35
    # UH(4) = 0.45

    # Implementation of the unit hydrograph calculation
    # (code from MATLAB file converted to Python)

    # Time step

    # Implement the unit hydrograph calculation here

    # Placeholder return value
    return np.array([[0.04, 0.17, 0.35, 0.45], [0, 0, 0, 0]])
