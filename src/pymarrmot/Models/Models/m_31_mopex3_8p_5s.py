import numpy as np
from marrmot_model import MARRMoT_model
from Models.Flux import (snowfall_1, rainfall_1, melt_1, evap_7, saturation_1,
                         recharge_3, baseflow_1)
class m_31_mopex3_8p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: MOPEX-3

    Model reference:
    Ye, S., Yaeger, M., Coopersmith, E., Cheng, L., & Sivapalan, M. (2012).
    Exploring the physical controls of regional patterns of flow duration
    curves - Part 2: Role of seasonality, the regime curve, and associated
    process controls. Hydrology and Earth System Sciences, 16(11), 4447–4465.
    http://doi.org/10.5194/hess-16-4447-2012
    """

    def __init__(self):
        super().__init__()
        self.numStores = 5  # number of model stores
        self.numFluxes = 11  # number of model fluxes
        self.numParams = 8
        self.JacobPattern = np.array([[1, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0],
                                      [0, 1, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 1, 0, 1]])  # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[-3, 3],  # tcrit, Snowfall & snowmelt temperature [oC]
                                   [0, 20],  # ddf, Degree-day factor for snowmelt [mm/oC/d]
                                   [1, 2000],  # Sb1, Maximum soil moisture storage [mm]
                                   [0, 1],  # tw, Groundwater leakage time [d-1]
                                   [0, 1],  # tu, Slow flow routing response time [d-1]
                                   [0.05, 0.95],  # se, Root zone storage capacity as fraction of Sb2 [-]
                                   [1, 2000],  # Sb2, Root zone storage capacity [mm]
                                   [0, 1]])  # tc, Mean residence time [d-1]

        self.StoreNames = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.FluxNames = ["ps", "pr", "qn", "et1", "q1f",
                          "qw", "et2", "q2f", "q2u", "qf", "qs"]  # Names for the fluxes

        self.FluxGroups = {"Ea": [4, 7],  # Index or indices of fluxes to add to Actual ET
                           "Q": [10, 11]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        pass

    def model_fun(self, S):
        theta = self.theta
        tcrit = theta[0]  # Snowfall & snowmelt temperature [oC]
        ddf = theta[1]  # Degree-day factor for snowmelt [mm/oC/d]
        s2max = theta[2]  # Maximum soil moisture storage [mm]
        tw = theta[3]  # Groundwater leakage time [d-1]
        tu = theta[4]  # Slow flow routing response time [d-1]
        se = theta[5]  # Root zone storage capacity as fraction of s3max [-]
        s3max = theta[6]  # Maximum groundwater storage [mm]
        tc = theta[7]  # Mean residence time [d-1]

        delta_t = self.delta_t

        S1, S2, S3, S4, S5 = S

        t = self.t  # this time step
        climate_in = self.input_climate(t)  # climate at this step
        P, Ep, T = climate_in

        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
        flux_et1 = evap_7(S2, s2max, Ep, delta_t)
        flux_q1f = saturation_1(flux_pr + flux_qn, S2, s2max)
        flux_qw = recharge_3(tw, S2)
        flux_et2 = evap_7(S3, se * s3max, Ep, delta_t)
        flux_q2f = saturation_1(flux_qw, S3, s3max)
        flux_q2u = baseflow_1(tu, S3)
        flux_qf = baseflow_1(tc, S4)
        flux_qs = baseflow_1(tc, S)

        # Stores ODEs
        dS1 = flux_ps - flux_qn
        dS2 = flux_pr + flux_qn - flux_et1 - flux_q1f - flux_qw
        dS3 = flux_qw - flux_et2 - flux_q2f - flux_q2u
        dS4 = flux_q1f + flux_q2f - flux_qf
        dS5 = flux_q2u - flux_qs

        # Outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = [flux_ps, flux_pr, flux_qn, flux_et1,
                  flux_q1f, flux_qw, flux_et2, flux_q2f,
                  flux_q2u, flux_qf, flux_qs]
        return dS, fluxes

    def step(self):
        pass

