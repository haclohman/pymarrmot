import numpy as np
from scipy.optimize import fsolve, lsqnonlin
from scipy.sparse import csr_matrix

class MARRMoT_model:
    """
    Superclass for all MARRMoT models
    """
    
    def __init__(self):
        #static attributes, set for each models in the model definition
        self.numStores = None           #number of model stores
        self.numFluxes = None           #number of model fluxes
        self.numParams = None           #number of model parameters
        self.parRanges = None           #default parameter ranges
        self.JacobPattern = None        #pattern of the Jacobian matrix of model store ODEs
        self.StoreNames = None          #names for the stores
        self.FluxNames = None           #Names for the fluxes
        self.FluxGroups = None          #Grouping of fluxes (useful for water balance and output)
        self.StoreSigns = None          #Signs to give to stores (-1 is a deficit store), assumes all 1 if not given
                                        
        #attributes set at the beginning of the simulation directly by the user
        self.theta = None               #Set of parameters
        self.delta_t = None             #time step
        self.S0 = None                  #initial store values
        self.input_climate = None       #vector of input climate
        self.solver_opts = None         #options for numerical solving of ODEs
                                        #automatically, based on parameter set
        self.store_min = None           #store minimum values
        self.store_max = None           #store maximum values

        #attributes created and updated automatically throughout a simulation
        self.t = None                   #current timestep
        self.fluxes = None              #vector of all fluxes
        self.stores = None              #vector of all stores
        self.uhs = None                 #unit hydrographs and still-to-flow fluxes
        self.solver_data = None         #step-by-step info of solver used and residuals
        self.status = None              #0 = model created, 1 = simulation ended
    
    def __setattr__(self, name, value):
        if name == 'delta_t':
            if isinstance(value, (int, float)) or value is None:
                self.delta_t = value
                self.reset()
            else:
                raise ValueError('delta_t must be a scalar')
        elif name == 'theta':
            if isinstance(value, np.ndarray) and value.size == self.numParams or value is None:
                self.theta = value.reshape(-1, 1)
                self.reset()
            else:
                raise ValueError(f'theta must have {self.numParams} elements')
        elif name == 'input_climate':
            if isinstance(value, np.ndarray):
                if value.shape[1] == 3:
                    self.input_climate = value
                    self.reset()
                else:
                    raise ValueError('Input climate must have 3 columns: precip, pet, temp')
            elif isinstance(value, dict):
                if all(key in value for key in ['precip', 'pet', 'temp']):
                    P = value['precip'] / self.delta_t
                    Ea = value['pet'] / self.delta_t
                    T = value['temp'] / self.delta_t
                    self.input_climate = np.column_stack((P, Ea, T))
                    self.reset()
                else:
                    raise ValueError('Input climate struct must contain fields: precip, pet, temp')
            elif value is None:
                self.input_climate = value
                self.reset()
            else:
                raise ValueError('Input climate must either be a dict or a numpy array of shape (n, 3)')
        elif name == 'S0':
            if isinstance(value, np.ndarray) and value.size == self.numStores or value is None:
                self.S0 = value.reshape(-1, 1)
                self.reset()
            else:
                raise ValueError(f'S0 must have {self.numStores} elements')
        elif name == 'solver_opts':
            if value is None:
                self.solver_opts = value
            elif isinstance(value, dict):
                self.solver_opts = self.add_to_def_opts(value)
                self.reset()
            else:
                raise ValueError('solver_opts must be a dictionary')
        else:
            super().__setattr__(name, value)

    #RESET is called any time that a user-specified input is changed
    #(t, delta_t, input_climate, S0, solver_options) and resets any
    #previous simulation ran on the object.
    #This is to prevent human error in analysing results.
    def reset(self):
        self.t = None
        self.fluxes = None
        self.stores = None
        self.uhs = None
        self.solver_data = None
        self.status = None

    def init_(self):
        self.store_min = np.zeros((self.numStores, 1))
        self.store_max = np.inf * np.ones((self.numStores, 1))
        t_end = self.input_climate.shape[0]
        self.stores = np.zeros((t_end, self.numStores))
        self.fluxes = np.zeros((t_end, self.numFluxes))
        self.solver_data = {
            'resnorm': np.zeros(t_end),
            'solver': np.zeros(t_end),
            'iter': np.zeros(t_end)
        }
        self.init()

    def ODE_approx_IE(self, S):
        S = S.reshape(-1, 1)
        delta_S = self.model_fun(S)
        Sold = self.S0 if self.t == 1 else self.stores[self.t - 1].reshape(-1, 1)
        return (S - Sold) / self.delta_t - delta_S

    def solve_stores(self, Sold):
        solver_opts = self.solver_opts
        resnorm_tolerance = solver_opts['resnorm_tolerance'] * (np.min(np.abs(Sold)) + 1e-5)
        resnorm_maxiter = solver_opts['resnorm_maxiter']

        Snew_v = np.zeros((3, self.numStores))
        resnorm_v = np.inf * np.ones(3)
        iter_v = np.ones(3)

        tmp_Snew, tmp_fval = fsolve(self.ODE_approx_IE, Sold, full_output=True)[:2]
        tmp_resnorm = np.sum(tmp_fval**2)
        Snew_v[0] = tmp_Snew.reshape(-1)
        resnorm_v[0] = tmp_resnorm

        if tmp_resnorm > resnorm_tolerance:
            tmp_Snew, tmp_fval, _, tmp_info = fsolve(self.ODE_approx_IE, tmp_Snew, full_output=True)[:4]
            tmp_resnorm = np.sum(tmp_fval**2)
            Snew_v[1] = tmp_Snew.reshape(-1)
            resnorm_v[1] = tmp_resnorm
            iter_v[1] = tmp_info['nfev']

            if tmp_resnorm > resnorm_tolerance:
                tmp_Snew, tmp_fval, _, tmp_info = lsqnonlin(self.ODE_approx_IE, tmp_Snew, bounds=(self.store_min.flatten(), self.store_max.flatten()), full_output=True)[:4]
                tmp_resnorm = np.sum(tmp_fval**2)
                Snew_v[2] = tmp_Snew.reshape(-1)
                resnorm_v[2] = tmp_resnorm
                iter_v[2] = tmp_info['nfev']

        best_iter = np.argmin(resnorm_v)
        Snew = Snew_v[best_iter].reshape(-1, 1)
        resnorm = resnorm_v[best_iter]
        iter_ = iter_v[best_iter]

        return Snew, resnorm, iter_

    def solve_fluxes(self, Sold):
        fluxes = np.zeros((self.numFluxes, 1))
        fluxes[self.uhs] = Sold[self.uhs] / self.theta[self.uhs]
        return fluxes

    def integrate_store(self, Sold):
        self.stores[self.t] = Sold.reshape(-1)
        self.fluxes[self.t] = self.solve_fluxes(Sold).reshape(-1)

    def integrate_one_step(self):
        Sold = self.stores[self.t - 1].reshape(-1, 1)
        Snew, resnorm, iter_ = self.solve_stores(Sold)
        self.solver_data['resnorm'][self.t] = resnorm
        self.solver_data['iter'][self.t] = iter_
        self.integrate_store(Snew)

    def integrate(self):
        self.init_()
        for self.t in range(1, self.input_climate.shape[0]):
            self.integrate_one_step()

    def get_fluxes(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.fluxes

    def get_stores(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.stores

    def get_sim_data(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.stores, self.fluxes

    def add_to_def_opts(self, new_opts):
        default_opts = {
            'resnorm_tolerance': 1e-5,
            'resnorm_maxiter': 1000
        }
        default_opts.update(new_opts)
        return default_opts

    def check_input(self):
        assert self.numStores == len(self.StoreNames), 'numStores must equal length of StoreNames'
        assert self.numFluxes == len(self.FluxNames), 'numFluxes must equal length of FluxNames'
        assert self.numParams == len(self.parRanges), 'numParams must equal length of parRanges'
        assert self.numParams == len(self.JacobPattern), 'numParams must equal length of JacobPattern'

    def jacob_fun(self, S):
        S = S.reshape(-1, 1)
        dfdS = self.model_jacob(S)
        return csr_matrix(dfdS)

    def solve_stores_nl(self, Sold):
        solver_opts = self.solver_opts
        resnorm_tolerance = solver_opts['resnorm_tolerance']
        resnorm_maxiter = solver_opts['resnorm_maxiter']

        resnorm_fun = lambda S: np.sum(self.ODE_approx_IE(S)**2)
        dfdS_fun = lambda S: self.jacob_fun(S)

        Snew, info = lsqnonlin(resnorm_fun, Sold, jac=dfdS_fun, bounds=(self.store_min.flatten(), self.store_max.flatten()), ftol=resnorm_tolerance, max_nfev=resnorm_maxiter, full_output=True)[:2]

        resnorm = np.sum(info['fvec']**2)
        iter_ = info['nfev']

        return Snew.reshape(-1, 1), resnorm, iter_

    def integrate_one_step_nl(self):
        Sold = self.stores[self.t - 1].reshape(-1, 1)
        Snew, resnorm, iter_ = self.solve_stores_nl(Sold)
        self.solver_data['resnorm'][self.t] = resnorm
        self.solver_data['iter'][self.t] = iter_
        self.integrate_store(Snew)

    def integrate_nl(self):
        self.init_()
        for self.t in range(1, self.input_climate.shape[0]):
            self.integrate_one_step_nl()

    def get_sim_data_nl(self):
        if self.t is None:
            self.integrate_nl()
        return self.stores, self.fluxes

    def model_fun(self, S):
        raise NotImplementedError

    def model_jacob(self, S):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError