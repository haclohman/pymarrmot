import numpy as np
from models.marrmot_model import MARRMoT_model
from models.flux import (evap_1, saturation_3, split_1, interception_1,
                         evap_3, percolation_2, baseflow_1)
from models.unit_hydro import (route, uh_3_half, update_uh)

class FlexiModel(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Flex-I

    Model reference:
    Fenicia, F., McDonnell, J. J., & Savenije, H. H. G. (2008). Learning from
    model improvement: On the contribution of complementary data to process 
    understanding. Water Resources Research, 44(6), 1–13. 
    http://doi.org/10.1029/2007WR006386
    """
    def __init__(self):
        super().__init__()
        self.numStores = 4                                              # number of model stores
        self.numFluxes = 11                                             # number of model fluxes
        self.numParams = 10 

        self.JacobPattern = np.array([[1, 0, 0, 0],
                                      [1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [1, 1, 0, 1]])                   # Jacobian matrix of model store ODEs
                              
        self.parRanges = np.array([[1, 2000],                           # URmax, Maximum soil moisture storage [mm]
                                    [0, 10],                             # beta, Unsaturated zone shape parameter [-]
                                    [0, 1],                              # D, Fast/slow runoff distribution parameter [-]
                                    [0, 20],                             # PERCmax, Maximum percolation rate [mm/d]
                                    [0.05, 0.95],                        # Lp, Wilting point as fraction of s1max [-]
                                    [1, 5],                              # Nlagf, Flow delay before fast runoff [d]
                                    [1, 15],                             # Nlags, Flow delay before slow runoff [d]
                                    [0, 1],                              # Kf, Fast runoff coefficient [d-1]
                                    [0, 1],                              # Ks, Slow runoff coefficient [d-1]
                                    [0, 5]])                             # Imax, Maximum interception storage [mm]
            
        self.StoreNames = ["S1", "S2", "S3", "S4"]                       # Names for the stores
        self.FluxNames = ["peff", "ei", "ru",  "eur", "ps",
                          "rf",   "rs", "rfl", "rsl", "qf", "qs"]        # Names for the fluxes
            
        self.FluxGroups = {"Ea": [2, 4],                                 # Index or indices of fluxes to add to Actual ET
                           "Q": [10, 11]}                                # Index or indices of fluxes to add to Streamflow

        self.uhs = None

    def init(self):
        """
        Initialization function.
        """
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        nlagf = theta[5]     # Flow delay before fast runoff [d]
        nlags = theta[6]     # Flow delay before slow runoff [d]
            
        # initialise the unit hydrographs and still-to-flow vectors            
        uh_f = uh_3_half(nlagf, delta_t)
        uh_s = uh_3_half(nlags, delta_t)
            
        self.uhs = [uh_f, uh_s]

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.

        Parameters:
        -----------
        S : numpy.ndarray
            State variables.

        Returns:
        --------
        tuple
            State derivatives and fluxes.
        """
        # parameters
        theta = self.theta
        smax, beta, d, percmax, lp, _, _, kf, ks, imax = theta

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uh_f, uh_s = self.uhs

        # stores
        S1, S2, S3, S4 = S

        # climate input
        t = self.t                                 # this time step
        climate_in = self.input_climate[t, :]      # climate at this step
        P, Ep, _ = climate_in

        # fluxes functions
        flux_peff = interception_1(P, S1, imax)
        flux_ei = evap_1(S1, Ep, delta_t)
        flux_ru = saturation_3(S2, smax, beta, flux_peff)
        flux_eur = evap_3(lp, S2, smax, Ep, delta_t)
        flux_ps = percolation_2(percmax, S2, smax, delta_t)
        flux_rf = split_1(1 - d, flux_peff - flux_ru)
        flux_rs = split_1(d, flux_peff - flux_ru)
        flux_rfl = route(flux_rf, uh_f)
        flux_rsl = route(flux_ps + flux_rs, uh_s)
        flux_qf = baseflow_1(kf, S3)
        flux_qs = baseflow_1(ks, S4)

        # stores ODEs
        dS1 = P - flux_peff - flux_ei
        dS2 = flux_ru - flux_eur - flux_ps
        dS3 = flux_rfl - flux_qf    
        dS4 = flux_rsl - flux_qs
            
        # outputs
        dS = np.array([dS1, dS2, dS3, dS4])
        fluxes = np.array([flux_peff, flux_ei, flux_ru, flux_eur, flux_ps,
                           flux_rf, flux_rs, flux_rfl, flux_rsl, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep, use it to update
        still-to-flow vectors from unit hydrographs.
        """
        # unit hydrographs and still-to-flow vectors
        uh_f, uh_s = self.uhs

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_ps = fluxes[4]
        flux_rf = fluxes[5]
        flux_rs = fluxes[6]

        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_f = update_uh(uh_f, flux_rf)
        uh_s = update_uh(uh_s, flux_ps + flux_rs)

        self.uhs = [uh_f, uh_s]
