import numpy as np
from models.marrmot_model import MARRMoT_model
from models.flux import evap_7, saturation_1, excess_1, interflow_3, recharge_3

class m_19_australia_8p_3s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Australia model

    Model reference:
    Farmer, D., Sivapalan, M., & Jothityangkoon, C. (2003). Climate, soil,
    and vegetation controls upon the variability of water balance in
    temperate and semiarid landscapes: Downward approach to water balance
    analysis. Water Resources Research, 39(2).
    http://doi.org/10.1029/2001WR000328
    """

    def __init__(self):
        """
        Creator method
        """
        self.numStores = 3  # number of model stores
        self.numFluxes = 8  # number of model fluxes
        self.numParams = 8  # number of model parameters

        self.JacobPattern = np.array([[1, 1, 0],
                                       [1, 1, 0],
                                       [0, 1, 1]])  # Jacobian matrix of model store ODEs

        self.parRanges = np.array([[1, 2000],  # Sb, Maximum soil moisture storage [mm]
                                   [0.05, 0.95],  # phi, Porosity [-]
                                   [0.01, 1.00],  # Sfc, Wilting point as fraction of sb [-]
                                   [0, 1.00],  # alpha_ss, Subsurface flow constant [1/d]
                                   [1, 5],  # beta_ss, Subsurface non-linearity constant [-]
                                   [0, 1.00],  # k_deep, Groundwater recharge constant [d-1]
                                   [0, 1.00],  # alpha_bf, Groundwater flow constant [d-1]
                                   [1, 5]])  # beta_bf, Groundwater non-linearity constant [-]

        self.StoreNames = ["S1", "S2", "S3"]  # Names for the stores
        self.FluxNames = ["eus", "rg", "se", "esat",
                          "qse", "qss", "qr", "qbf"]  # Names for the fluxes

        self.FluxGroups = {"Ea": [1, 4],  # Index or indices of fluxes to add to Actual ET
                           "Q": [5, 6, 8]}  # Index or indices of fluxes to add to Streamflow

    def init(self):
        """
        INITialisation function
        """
        pass

    def model_fun(self, S):
        """
        MODEL_FUN are the model governing equations in state-space formulation

        Parameters:
        S : array-like
            State variables

        Returns:
        tuple: Tuple containing two arrays (dS, fluxes)
               dS: Array of derivatives of state variables
               fluxes: Array of fluxes
        """
        # parameters
        theta = self.theta
        sb = theta[0]  # Maximum soil moisture storage [mm]
        phi = theta[1]  # Porosity [-]
        fc = theta[2]  # Field capacity as fraction of sb [-]
        alpha_ss = theta[3]  # Subsurface flow constant [1/d]
        beta_ss = theta[4]  # Subsurface non-linearity constant [-]
        k_deep = theta[5]  # Groundwater recharge constant [d-1]
        alpha_bf = theta[6]  # Groundwater flow constant [d-1]
        beta_bf = theta[7]  # Groundwater non-linearity constant [-]

        # delta_t
        delta_t = self.delta_t

        # stores
        S1, S2, S3 = S

        # climate input
        t = self.t  # this time step
        climate_in = self.input_climate(t)  # climate at this step
        P, Ep = climate_in[0], climate_in[1]

        # fluxes functions
        flux_eus = evap_7(S1, sb, Ep, delta_t)
        flux_rg = saturation_1(P, S1, (sb - S2) * fc / phi)
        flux_se = excess_1(S1, (sb - S2) * fc / phi, delta_t)
        flux_esat = evap_7(S2, sb, Ep, delta_t)
        flux_qse = saturation_1(flux_rg + flux_se, S2, sb)
        flux_qss = interflow_3(alpha_ss, beta_ss, S2, delta_t)
        flux_qr = recharge_3(k_deep, S2)
        flux_qbf = interflow_3(alpha_bf, beta_bf, S3, delta_t)

        # stores ODEs
        dS1 = P - flux_eus - flux_rg - flux_se
        dS2 = flux_rg + flux_se - flux_esat - flux_qse - flux_qss - flux_qr
        dS3 = flux_qr - flux_qbf

        # outputs
        dS = np.array([dS1, dS2, dS3])
        fluxes = np.array([flux_eus, flux_rg, flux_se, flux_esat,
                           flux_qse, flux_qss, flux_qr, flux_qbf])

        return dS, fluxes

    def step(self):
        """
        STEP runs at the end of every timestep
        """
        pass
