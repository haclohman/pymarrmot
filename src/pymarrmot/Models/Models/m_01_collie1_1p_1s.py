from models.marrmot_model import MARRMoT_model
from models.flux import evap_7, saturation_1

class M_01_Collie1_1p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Collie River 1 (traditional bucket model)
    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development. Journal of Hydrology, 254,
    174–198. doi: 10.1016/S0022-1694(01)497-6.
    """
    def __init__(self):
        self.numStores = 1                  # number of model stores
        self.numFluxes = 2                  # number of model fluxes
        self.numParams = 1
        self.JacobPattern = [1]             # Jacobian matrix of model store ODEs
        self.parRanges = [1, 2000]          # Smax [mm]
        self.StoreNames = ["S1"]            # Names for the stores
        self.FluxNames = ["ea", "qse"]      # Names for the fluxes
        self.FluxGroups = {'Ea': 1, 'Q': 2} # Index or indices of fluxes to add to Actual ET and Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        # parameters
        theta = self.theta
        S1max = theta[0]  # Maximum soil moisture storage [mm]
        # delta_t
        delta_t = self.delta_t
        # stores
        S1 = S[0]
        # climate input
        t = self.t  # this time step
        climate_in = self.input_climate[t, :]  # climate at this step
        P = climate_in[0]
        Ep = climate_in[1]
        # fluxes functions
        flux_ea = evap_7(S1, S1max, Ep, delta_t)
        flux_qse = saturation_1(P, S1, S1max)
        # stores ODEs
        dS1 = P - flux_ea - flux_qse
        # outputs
        dS = [dS1]
        fluxes = [flux_ea, flux_qse]
        return dS, fluxes

    def step(self):
        pass


