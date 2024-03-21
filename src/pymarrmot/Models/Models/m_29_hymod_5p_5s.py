import numpy as np
from models.marrmot_model import MARRMoT_model
from models.flux import (evap_7, saturation_2, split_1, baseflow_1)

class m_29_hymod_5p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: HyMOD
    """

    def __init__(self):
        """
        Creator method.
        """
        self.numStores = 5                                           # number of model stores
        self.numFluxes = 8                                           # number of model fluxes
        self.numParams = 5

        self.JacobPattern = np.array([[1, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0],
                                       [0, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 0],
                                       [1, 0, 0, 0, 1]])             # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[1, 2000],                        # Smax, Maximum soil moisture storage [mm]
                                    [0, 10],                          # b, Soil depth distribution parameter [-]
                                    [0, 1],                           # a, Runoff distribution fraction [-]
                                    [0, 1],                           # kf, fast flow time parameter [d-1]
                                    [0, 1]])                          # ks, base flow time parameter [d-1]

        self.StoreNames = ["S1", "S2", "S3", "S4", "S5"]             # Names for the stores
        self.FluxNames = ["ea", "pe", "pf", "ps",
                          "qf1", "qf2", "qf3", "qs"]               # Names for the fluxes

        self.FluxGroups = {'Ea': 1, 'Q': [7, 8]}                     # Index or indices of fluxes to add to Actual ET and Streamflow

    def init(self):
        """
        Initialization function.
        """
        pass

    def model_fun(self, S):
        """
        Model governing equations in state-space formulation.
        """
        # parameters
        smax, b, a, kf, ks = self.theta

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3, S4, S5 = S

        # climate input
        t = self.t                                                   # this time step
        climate_in = self.input_climate[t, :]                        # climate at this step
        P, Ep, _ = climate_in

        # fluxes functions
        flux_ea = evap_7(S1, smax, Ep, delta_t)
        flux_pe = saturation_2(S1, smax, b, P)
        flux_pf = split_1(a, flux_pe)
        flux_ps = split_1(1 - a, flux_pe)
        flux_qf1 = baseflow_1(kf, S2)
        flux_qf2 = baseflow_1(kf, S3)
        flux_qf3 = baseflow_1(kf, S4)
        flux_qs = baseflow_1(ks, S5)

        # stores ODEs
        dS1 = P - flux_ea - flux_pe
        dS2 = flux_pf - flux_qf1
        dS3 = flux_qf1 - flux_qf2
        dS4 = flux_qf2 - flux_qf3
        dS5 = flux_ps - flux_qs

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = np.array([flux_ea, flux_pe, flux_pf, flux_ps,
                           flux_qf1, flux_qf2, flux_qf3, flux_qs])

        return dS, fluxes

    def step(self):
        """
        Runs at the end of every timestep.
        """
        pass
