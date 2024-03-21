
import numpy as np

def route(flux_in, uh):
    '''
    ROUTE calculates the output of a unit hydrograph at the current timestep
    after routing a flux through it.

    Parameters:
    flux_in : float
        Input flux

    uh : numpy array
        Unit hydrograph

    Returns:
    flux_out : numpy array
        Output flux
    '''
    # Perform the routing calculation
    flux_out = np.convolve(uh, flux_in, mode='full')[:len(uh)]
    
    return flux_out

# Example usage
flux_in = 1.5
uh = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
output_flux = route(flux_in, uh)
print("Output Flux:", output_flux)
