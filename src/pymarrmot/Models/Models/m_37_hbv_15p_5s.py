import numpy as np
from models.marrmot_model import MARRMoT_model
from models.unit_hydro import(uh_4_full, update_uh)

class m_37_hbv_15p_5s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: HBV-96

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S.
    (1997). Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201, 272–288.
    https://doi.org/https://doi.org/10.1016/S0022-1694(97)00041-3
    """

    def __init__(self):
        """
        Creator method
        """
        self.num_stores = 5  # number of model stores
        self.num_fluxes = 13  # number of model fluxes
        self.num_params = 15

        self.jacob_pattern = np.array([[1, 1, 0, 0, 0],
                                       [1, 1, 0, 0, 0],
                                       [1, 1, 1, 1, 0],
                                       [1, 1, 1, 1, 0],
                                       [0, 0, 0, 1, 1]])  # Jacobian matrix of model store ODEs

        self.par_ranges = np.array([[-3, 5],      # TT, threshold temperature for snowfall [oC]
                                    [0, 17],       # TTI, interval length of rain-snow spectrum [oC]
                                    [-3, 3],       # TTM, threshold temperature for snowmelt [oC]
                                    [0, 1],        # CFR, coefficient of refreezing of melted snow [-]
                                    [0, 20],       # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
                                    [0, 1],        # WHC, maximum water holding content of snow pack [-]
                                    [0, 4],        # CFLUX, maximum rate of capillary rise [mm/d]
                                    [1, 2000],     # FC, maximum soil moisture storage [mm]
                                    [0.05, 0.95],  # LP, wilting point as fraction of FC [-]
                                    [0, 10],       # BETA, non-linearity coefficient of upper zone recharge [-]
                                    [0, 1],        # K0, runoff coefficient from upper zone [d-1]
                                    [0, 4],        # ALPHA, non-linearity coefficient of runoff from upper zone [-]
                                    [0, 20],       # PERC, maximum rate of percolation to lower zone [mm/d]
                                    [0, 1],        # K1, runoff coefficient from lower zone [d-1]
                                    [1, 120]])     # MAXBAS, flow routing delay [d]

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]  # Names for the stores
        self.flux_names = ["sf", "refr", "melt", "rf", "in", "se", "cf",
                           "ea", "r", "q0", "perc", "q1", "qt"]  # Names for the fluxes

        self.flux_groups = {"Ea": 7,  # Index or indices of fluxes to add to Actual ET
                            "Q": 12}   # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        # parameters
        theta = self.theta
        delta_t = self.delta_t
        maxbas = theta[14]  # MAXBAS, flow routing delay [d]

        # initialise the unit hydrographs and still-to-flow vectors
        uh = uh_4_full(maxbas, delta_t)
        self.uhs = [uh]

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation
        """
        # parameters
        theta = self.theta
        tt = theta[0]      # TT, middle of snow-rain interval [oC]
        tti = theta[1]     # TTI, interval length of rain-snow spectrum [oC]
        ttm = theta[2]     # TTM, threshold temperature for snowmelt [oC]
        cfr = theta[3]     # CFR, coefficient of refreezing of melted snow [-]
        cfmax = theta[4]   # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
        whc = theta[5]     # WHC, maximum water holding content of snow pack [-]
        cflux = theta[6]   # CFLUX, maximum rate of capillary rise [mm/d]
        fc = theta[7]      # FC, maximum soil moisture storage [mm]
        lp = theta[8]      # LP, wilting point as fraction of FC [-]
        beta = theta[9]    # BETA, non-linearity coefficient of upper zone recharge [-]
        k0 = theta[10]     # K0, runoff coefficient from upper zone [d-1]
        alpha = theta[11]  # ALPHA, non-linearity coefficient of runoff from upper zone [-]
        perc = theta[12]   # PERC, maximum rate of percolation to lower zone [mm/d]
        k1 = theta[13]     # K1, runoff coefficient from lower zone [d-1]
        maxbas = theta[14] # MAXBAS, flow routing delay [d]

        # delta_t
        delta_t = self.delta_t

        # unit hydrographs and still-to-flow vectors
        uh = self.uhs[0]

        # stores
        S1, S2, S3, S4, S5 = S

        # stores at previous timestep
        t = self.t
        if t == 1:
            S2old = self.S0[1]
        else:
            S2old = self.stores[t - 1, 1]

        # climate input
        climate_in = self.input_climate(t, :)


    def step(self):
        """
        STEP runs at the end of every timestep, use it to update
        still-to-flow vectors from unit hydrographs
        """
        # unit hydrographs and still-to-flow vectors
        uh = self.uhs[0]

        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_q0 = fluxes[9]
        flux_q1 = fluxes[11]

        # update still-to-flow vectors using
        # fluxes at current step and unit hydrographs
        uh = update_uh(uh, flux_q0 + flux_q1)
        self.uhs = [uh]
