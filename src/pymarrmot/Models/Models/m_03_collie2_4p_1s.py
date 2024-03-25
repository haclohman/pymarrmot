from pymarrmot.models.models.marrmot_model import MARRMoT_model
from pymarrmot.models.flux import (evap_7, evap_3, saturation_1, interflow_8)

class m_03_collie2_4p_1s(MARRMoT_model):
    """
    Class for hydrologic conceptual model: Collie River v2

    Copyright (C) 2019, 2021 Wouter J.M. Knoben, Luca Trotter
    This file is part of the Modular Assessment of Rainfall-Runoff Models
    Toolbox (MARRMoT).
    MARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
    WARRANTY. See <https://www.gnu.org/licenses/> for details.

    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), 'Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development.' Journal of Hydrology, 254,
    174-198. doi: 10.1016/S0022-1694(01)00496-6.
    """

    def __init__(self):
        """
        Initialize the Collie River v2 model.
        """
        super().__init__()
        self.numStores = 1
        self.numFluxes = 4
        self.numParams = 4

        self.JacobPattern = [1]

        self.parRanges = [
            [1, 2000],     # Smax [mm]
            [0.05, 0.95],  # fc as fraction of Smax
            [0, 1],        # a, subsurface runoff coefficient [d-1]
            [0.05, 0.95]   # M, fraction forest cover [-]
        ]

        self.StoreNames = ["S1"]
        self.FluxNames = ["eb", "ev", "qse", "qss"]

        self.FluxGroups = {
            'Ea': [1, 2],  # Index or indices of fluxes to add to Actual ET
            'Q': [3, 4]    # Index or indices of fluxes to add to Streamflow
        }

    def init(self):
        """
        Initialize the model.
        """
        pass

    def model_fun(self, S):
        """
        Calculate model dynamics.

        Parameters:
        S (array_like): State variables.

        Returns:
        dS (array_like): Derivative of state variables.
        fluxes (array_like): Model fluxes.
        """
        theta = self.theta
        S1max = theta[0]  # Maximum soil moisture storage [mm]
        Sfc = theta[1]    # Field capacity as fraction of S1max [-]
        a = theta[2]      # Subsurface runoff coefficient [d-1]
        M = theta[3]      # Fraction forest cover [-]

        delta_t = self.delta_t

        S1 = S[0]

        t = self.t
        climate_in = self.input_climate[t, :]
        P = climate_in[0]
        Ep = climate_in[1]
        T = climate_in[2]

        flux_eb = evap_7(S1, S1max, (1 - M) * Ep, delta_t)
        flux_ev = evap_3(Sfc, S1, S1max, M * Ep, delta_t)
        flux_qse = saturation_1(P, S1, S1max)
        flux_qss = interflow_8(S1, a, Sfc * S1max)

        dS1 = P - flux_eb - flux_ev - flux_qse - flux_qss

        dS = [dS1]
        fluxes = [flux_eb, flux_ev, flux_qse, flux_qss]

        return dS, fluxes

    def step(self):
        """
        Perform model step.
        """
        pass