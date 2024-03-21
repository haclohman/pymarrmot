import numpy as np
from marrmot_model import MARRMoT_model
from Models.UnitHydro import (uh_3_half, route, update_uh)
from Models.Flux import (snowfall_1, rainfall_1, melt_1, 
    interception_1, evap_1, saturation_3, evap_3, percolation_2, split_1, baseflow_1)

class M34FlexIS(MARRMoT_model):
    def __init__(self):
        super().__init__()
        self.numStores = 5                                             # number of model stores
        self.numFluxes = 14                                            # number of model fluxes
        self.numParams = 12
            
        self.JacobPattern = np.array([[1, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0],
                                      [1, 1, 1, 0, 0],
                                      [1, 1, 1, 1, 0],
                                      [1, 1, 1, 0, 1]])              # Jacobian matrix of model store ODEs
                             
        self.parRanges = np.array([[1, 2000],       # URmax, Maximum soil moisture storage [mm]
                                    [0, 10],         # beta, Unsaturated zone shape parameter [-]
                                    [0, 1],          # D, Fast/slow runoff distribution parameter [-]
                                    [0, 20],         # PERCmax, Maximum percolation rate [mm/d]
                                    [0.05, 0.95],    # Lp, Wilting point as fraction of s1max [-]
                                    [1, 5],          # Nlagf, Flow delay before fast runoff [d]
                                    [1, 15],         # Nlags, Flow delay before slow runoff [d]
                                    [0, 1],          # Kf, Fast runoff coefficient [d-1]
                                    [0, 1],          # Ks, Slow runoff coefficient [d-1]
                                    [0, 5],          # Imax, Maximum interception storage [mm]
                                    [-3, 5],         # TT, Threshold temperature for snowfall/snowmelt [oC]
                                    [0, 20]])        # ddf, Degree-day factor for snowmelt [mm/d/oC]
            
        self.StoreNames = ["S1", "S2", "S3", "S4", "S5"]                   # Names for the stores
        self.FluxNames  = ["ps", "pi", "m", "peff", "ei",
                           "ru", "eur", "rp", "rf", "rs",
                           "rf1", "rs1", "qf", "qs"]                     # Names for the fluxes
            
        self.FluxGroups = {'Ea': [5, 7],                                   # Index or indices of fluxes to add to Actual ET
                           'Q': [13, 14]}                                  # Index or indices of fluxes to add to Streamflow
            
    def init(self):
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
        # parameters
        theta = self.theta
        smax = theta[0]     # Maximum soil moisture storage [mm]
        beta = theta[1]     # Unsaturated zone shape parameter [-]
        d = theta[2]        # Fast/slow runoff distribution parameter [-]
        percmax = theta[3]  # Maximum percolation rate [mm/d]
        lp = theta[4]       # Wilting point as fraction of s1max [-]
        nlagf = theta[5]    # Flow delay before fast runoff [d]
        nlags = theta[6]    # Flow delay before slow runoff [d]
        kf = theta[7]       # Fast runoff coefficient [d-1]
        ks = theta[8]       # Slow runoff coefficient [d-1]
        imax = theta[9]     # Maximum interception storage [mm]
        tt = theta[10]      # Threshold temperature for snowfall/snowmelt [oC]
        ddf = theta[11]     # Degree-day factor for snowmelt [mm/d/oC]
            
        # delta_t
        delta_t = self.delta_t
            
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_f = uhs[0]
        uh_s = uhs[1]
            
        # stores
        S1 = S[0]
        S2 = S[1]
        S3 = S[2]
        S4 = S[3]
        S5 = S[4]
            
        # climate input
        t = self.t                             # this time step
        climate_in = self.input_climate[t, :]   # climate at this step
        P = climate_in[0]
        Ep = climate_in[1]
        T = climate_in[2]
            
        # fluxes functions
        flux_ps = snowfall_1(P, T, tt)
        flux_pi = rainfall_1(P, T, tt)
        flux_m = melt_1(ddf, tt, T, S1, delta_t)
        flux_peff = interception_1(flux_m + flux_pi, S2, imax)
        flux_ei = evap_1(S2, Ep, delta_t)
        flux_ru = saturation_3(S3, smax, beta, flux_peff)
        flux_eur = evap_3(lp, S3, smax, Ep, delta_t)
        flux_rp = percolation_2(percmax, S3, smax, delta_t)
        flux_rf = split_1(1 - d, flux_peff - flux_ru)
        flux_rs = split_1(d, flux_peff - flux_ru)
        flux_rfl = route(flux_rf, uh_f)
        flux_rsl = route(flux_rs + flux_rp, uh_s)
        flux_qf = baseflow_1(kf, S4)
        flux_qs = baseflow_1(ks, S5)
            
        # stores ODEs
        dS1 = flux_ps - flux_m
        dS2 = flux_m + flux_pi - flux_peff - flux_ei
        dS3 = flux_ru - flux_eur - flux_rp
        dS4 = flux_rfl - flux_qf
        dS5 = flux_rsl - flux_qs 

        # outputs
        dS = np.array([dS1, dS2, dS3, dS4, dS5])
        fluxes = np.array([flux_ps, flux_pi, flux_m, flux_peff, flux_ei,
                           flux_ru, flux_eur, flux_rp, flux_rf, flux_rs, 
                           flux_rfl, flux_rsl, flux_qf, flux_qs])

        return dS, fluxes

    def step(self):
        # unit hydrographs and still-to-flow vectors
        uhs = self.uhs
        uh_f = uhs[0]
        uh_s = uhs[1]
            
        # input fluxes to the unit hydrographs
        fluxes = self.fluxes[self.t, :]
        flux_rp = fluxes[7]
        flux_rf = fluxes[8]
        flux_rs = fluxes[9]
            
        # update still-to-flow vectors using fluxes at current step and
        # unit hydrographs
        uh_f = update_uh(uh_f, flux_rf)
        uh_s = update_uh(uh_s, flux_rs + flux_rp)
            
        self.uhs = [uh_f, uh_s]
