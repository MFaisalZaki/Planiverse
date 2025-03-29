# This is a refactored version of cG_LAGTPKS_Environment from https://github.com/fstrnad/pyDRLinWESM.git
"""
This Environment describes the 7D implementation of the copan:GLOBAL model developed by Jobst Heitzig.
The parameters are taken from Nitzbon et al. 2017. 
The code contains implementation parts that go back to Jan Nitzbon 2016 
Dynamic variables are :
    - terrestrial ("land") carbon L
    - excess atmospheric carbon stock A
    - geological carbon G
    - temperature T
    - population P
    - capital K
    - the renewable energy knowledge stock S

Parameters (mainly Nitzbon et al. 2016 )
----------
    - sim_time: Timestep that will be integrated in this simulation step
        In each grid point the agent can choose between subsidy None, A, B or A and B in combination. 
    - Sigma = 1.5 * 1e8
    - CstarPI=4000
    - Cstar=5500
    - a0=0.03
    - aT=3.2*1e3
    - l0=26.4
    - lT=1.1*1e6
    - delta=0.01
    - m=1.5
    - g=0.02
    - p=0.04
    - Wp=2000
    - q0=20
    - b=5.4*1e-7
    - yE=147
    - eB=4*1e10
    - eF=4*1e10
    - i=0.25
    - k0=0.1
    - aY=0.
    - aB= 3e5 (varied, basic year 2000)
    - aF= 5e6 (varied, basic year 2000)
    - aR= 7e-18 (varied, basic year 2000)
    - sS=1./50.
    - sR=1.
"""

import numpy as np
from scipy.integrate import odeint

default_simulation_parameters = dict(
    Sigma = 1.5 * 1e8,
    Cstar=5500,
    a0=0.03,
    aT=3.2*1e3,
    l0=26.4,
    lT=1.1*1e6,
    delta=0.01,
    m=1.5,
    g=0.02,
    p=0.04,
    Wp=2000,
    q0=20,
    qP=0.,
    b=5.4*1e-7,
    yE=120,
    wL=0.,
    eB=4*1e10,
    eF=4*1e10,
    i=0.25,
    k0=0.1,
    aY=0.,
    aB=1.5e4,
    aF=2.7e5,
    aR=9e-15,
    sS=1./50.,
    sR=1.,
    ren_sub=.5,
    carbon_tax=.5,
    i_DG=0.1, 
    L0=0.3*2480,
)

default_state_parameters=dict(L=2480., A=830.0, G=1125, T=5.053333333333333e-6, P=6e9, K=5e13, S=5e11)

@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)

@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)

class WorldState:
    """
    We define the state regarding those variables:
        - terrestrial ("land") carbon L
        - excess atmospheric carbon stock A
        - geological carbon G
        - temperature T
        - population P
        - capital K
        - the renewable energy knowledge stock S
    """
    def __init__(self, attrs, action, t):
        self.parameter_names =['land_carbon', 'atmospheric_carbon', 'geological_carbon', 'temperature', 'population', 'capital', 'renewable_energy']
        self.state = np.array([attrs['L'], attrs['A'], attrs['G'], attrs['T'], attrs['P'], attrs['K'], attrs['S']])
        self.t = t
        self.action = action
        self.literals = frozenset([])
        self.__update__()

    def __eq__(self, value):
        return all(self.state == value.state)

    def __update__(self):
        self.literals = frozenset([f'parameter_{name}_{val}'.replace('e-','e_n').replace('.','_') for name, val in zip(self.parameter_names, self.state)])

class WorldAction:
    def __init__(self, action):
        # Tuple: of three actions:
        # [0]: 'Sub'
        # [1]: 'Tax'
        # [2]: 'NP'
        self.action = action

class WorldSimulatorEnv:
    
    def __init__(self, simulation_parameters=default_simulation_parameters, initial_state_params=default_state_parameters, specs=[]):
        self.t                     = None
        self.dt                    = 1
        self.current_state         = None
        self.ini_state             = None
        self.simulation_parameters = simulation_parameters
        self.initial_state_params  = initial_state_params
        self.specs                 = specs

        self.actionslist = [
            WorldAction((False, False, False)), 
            WorldAction((False, False, True)),
            WorldAction((False, True,  False)), 
            WorldAction((False, True,  True)),
            WorldAction((True,  False, False)), 
            WorldAction((True,  False, True)),
            WorldAction((True,  True,  False)), 
            WorldAction((True,  True,  True))
        ]
        

        self.__check__init_parameters__()
        self.__compuate_derived_variables__()
        self.__set_planetary_boundaries__()

    def __check__init_parameters__(self):
        assert 0 <= self.simulation_parameters['Cstar'], "Cstar must be non-negative"
        for p in ['L', 'A', 'G', 'T', 'P', 'K', 'S']:
            assert 0 <= self.initial_state_params[p], f"{p} must be non-negative"
        
        for p in ['L', 'A', 'G']:
            assert self.initial_state_params[p] <= self.simulation_parameters['Cstar'], f"{p} must be <= Cstar"
        
        for p in ['L', 'A']:
            assert self.initial_state_params[p] <= self.simulation_parameters['Cstar'] - self.initial_state_params[p], f"{p} must be <= Cstar - G"

    def __compuate_derived_variables__(self):

        Xb = self.simulation_parameters['aB'] * self.initial_state_params['L']**2.
        Xf = self.simulation_parameters['aF'] * self.initial_state_params['G']**2.
        Xr = self.simulation_parameters['aR'] * self.initial_state_params['S']**2.
        expP = 2. / 5.
        expK = 2. / 5.

        Z = self.Z(self.initial_state_params['P'], self.initial_state_params['K'], Xb + Xf + Xr, expP, expK)

        Aini = self.initial_state_params['A']
        Pini = self.initial_state_params['P']
        Bini = self.B(Xb, Z)
        Fini = self.F(Xf, Z)
        Rini = self.R(Xr, Z)
        Yini = self.Y(Bini, Fini, Rini)
        Wini = self.W(Yini, Pini, self.initial_state_params['L'])

        self.ini_state =  np.array([Aini, Wini, Pini])

    def __set_planetary_boundaries__(self):
        self.A_PB       = 945
        self.A_scale    = 1
        self.Y_PB       = self.direct_Y(self.initial_state_params['L'], self.initial_state_params['G'], self.initial_state_params['P'], self.initial_state_params['K'], self.initial_state_params['S'])
        self.P_PB       = 1e6
        self.W_PB       = (1- self.simulation_parameters['i'])*self.Y_PB / (1.01*self.initial_state_params['P'])   # Economic production in year 2000 and population in year 2000
        self.W_scale    = 1e3
        self.PB         = np.array([self.A_PB, self.W_PB, self.P_PB])
        self.compact_PB = compactification(self.PB, self.ini_state)    
        self.P_scale    = 1e9

    #economic production
    def Y(self, B, F, R): return self.simulation_parameters['yE'] * (self.simulation_parameters['eB']*B + self.simulation_parameters['eF']*F + R )
    #wellbeing
    def W(self, Y, P, L): return (1.-self.simulation_parameters['i']) * Y / P + self.simulation_parameters['wL']*L/self.simulation_parameters['Sigma']
    #auxiliary
    def Z(self, P, K, X, expP=2./5, expK=2./5.): return P**expP * K**expK / X**(4./5.)
    def B(self, Xb, Z): return Xb * Z / self.simulation_parameters['eB']
    def F(self, Xf, Z): return Xf * Z / self.simulation_parameters['eF']
    def R(self,Xr, Z): return Xr * Z 
    def direct_W(self,L,G,P,K,S): return self.W(self.direct_Y(L, G, P, K, S), P, L)

    def direct_Y(self, L,G,P,K,S):
        expP=2./5.
        expK=2./5.

        Xb=self.simulation_parameters['aB']*L**2.
        Xf=self.simulation_parameters['aF']*G**2.
        Xr=self.simulation_parameters['aR']*S**2.  
        X=Xb+Xf+Xr

        if 'KproptoP' in self.specs: K = P*self.initial_state_params['K']/(self.initial_state_params['P'])
        if 'NproptoP' in self.specs: expP-=1./5.
        Z=self.Z(P, K, X, expP, expK)           
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)
        return self.Y(B, F, R)
    
    def __successor_state__(self, state, action):
        # compute the updated actions parameter.
        traj_one_step=odeint(self.__simulate_dynamics__, state.state, [state.t, state.t+self.dt], args=self.__translate_action__(action), mxstep=50000)
        dL = traj_one_step[:,0][-1]
        dA = traj_one_step[:,1][-1]
        dG = traj_one_step[:,2][-1]
        dT = traj_one_step[:,3][-1]
        dP = traj_one_step[:,4][-1]
        dK = traj_one_step[:,5][-1]
        dS = traj_one_step[:,6][-1]
        return WorldState(dict(L=dL, A=dA, G=dG, T=dT, P=dP, K=dK, S=dS), action, state.t + self.dt)

    def __reward__(self, state, action):
        L,A,G,T,P,K,S=  state.state
        Leff = max(L-self.L0, 0) if state.action[-1] else L       
        W    = self.direct_W(Leff, G, P, K, S)
        return np.linalg.norm( self.compact_PB -  compactification( np.array([A, W, P]), self.ini_state)) if self.is_terminal(state) else 0.
    
    """
    This functions define the dynamics of the copan:GLOBAL model
    """
    def __simulate_dynamics__(self, LAGTPKS, dt, aR, aB, aF, Lprot):
        #auxiliary functions
        #photosynthesis
        def phot(L, A, T): return (self.simulation_parameters['l0']-self.simulation_parameters['lT']*T)*np.sqrt(A)/np.sqrt(self.simulation_parameters['Sigma'])
        #respiration
        def resp(L, T): return self.simulation_parameters['a0']+self.simulation_parameters['aT']*T
        #diffusion atmosphere <--> ocean
        def diff(L, A, G=0.): return self.simulation_parameters['delta']*(self.simulation_parameters['Cstar']-L-G-(1+self.simulation_parameters['m'])*A)
        def fert(P,W): return 2*self.simulation_parameters['p']*self.simulation_parameters['Wp']*W/(self.simulation_parameters['Wp']**2+W**2) 
        def mort(P,W): return self.simulation_parameters['q0']/(W) + self.simulation_parameters['qP']*P/self.simulation_parameters['Sigma']
        
        L, A, G, T, P, K, S = LAGTPKS
        #adjust to lower and upper bounds
        L=np.amin([np.amax([L, 1e-12]), self.simulation_parameters['Cstar']])
        A=np.amin([np.amax([A, 1e-12]), self.simulation_parameters['Cstar']])
        G=np.amin([np.amax([G, 1e-12]), self.simulation_parameters['Cstar']])
        T=np.amax([T, 1e-12])
        P=np.amax([P, 1e-12])
        K=np.amax([K, 1e-12])
        S=np.amax([S, 1e-12])

        # calculate T and A if instantaneous processes
        if 'INST_DIFF' in self.specs:
            A = (self.simulation_parameters['Cstar']-L-G) / (1.+self.simulation_parameters['m'])
        if 'INST_GH' in self.specs:
            T = A/self.simulation_parameters['Sigma']
        #calculate auxiliary quantities
        
        Leff = max(L-self.simulation_parameters['L0'], 0) if Lprot else L
        
        Xb = aB*Leff**2.
        Xf = aF*G**2.
        Xr = aR*S**2. 
        X = Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        if 'KproptoP' in self.specs: K = P*self.initial_state_params['K']/(self.initial_state_params['P'])
        if 'NproptoP' in self.specs: expP-=1./5.

        Z = self.Z(P, K, X, expP, expK)
        
        #calculate derived variables
        B = self.B(Xb, Z)
        F = self.F(Xf, Z)
        R = self.R(Xr, Z)

        Y = self.Y(B, F, R)
        W = self.W(Y, P, L)

        # calculate derivatives of the dynamic variables
        dL = (phot(L, A, T) - resp(L, T)) * L - B
        dA = -dL + diff(L, A, G)
        dG = -F
        dT = self.simulation_parameters['g'] * (A/self.simulation_parameters['Sigma'] - T)
        dP = P * (fert(P,W)-mort(P,W))
        dK = self.simulation_parameters['i'] * Y - self.simulation_parameters['k0'] * K
        dS = self.simulation_parameters['sR']*R - self.simulation_parameters['sS']*S

        if 'INST_DIFF' in self.specs: dA = -(dL+dG)/(1.+self.simulation_parameters['m'])
        if 'INST_GH' in self.specs:   dT = dA/self.simulation_parameters['Sigma']

        return [dL, dA, dG, dT, dP, dK, dS]
    
    def __translate_action__(self, action):
        """
        This function is needed to adjust the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen 
        management option.
        Parameters:
            -action: Number of the action in the actionset.
             Can be transformed into: 'default', 'subsidy' 'carbon tax' 'Nature Protection ' or possible combinations
        """
        a = action.action
        ret_values = []
        # subsidy 
        ret_values += [self.simulation_parameters['aR']*(1+self.simulation_parameters['ren_sub'])]    if a[0] else [self.simulation_parameters['aR']]
        # carbon tax
        ret_values += [self.simulation_parameters['aB']*(1-self.simulation_parameters['carbon_tax'])] if a[1] else [self.simulation_parameters['aB']]
        ret_values += [self.simulation_parameters['aF']*(1-self.simulation_parameters['carbon_tax'])] if a[1] else [self.simulation_parameters['aF']]
        # nature protection
        ret_values += [a[2]]
        # aR, aB, Af, Lprot
        return tuple(ret_values)

    def reset(self):
        self.t = 0
        self.current_state = WorldState(self.initial_state_params, self.actionslist[0], self.t)
        self.__compuate_derived_variables__()
        return self.current_state, {}
    
    def step(self, action):
        new_state = self.__successor_state__(self.current_state, action)
        terminal_state = self.is_goal(new_state) or self.is_terminal(new_state)
        reward = self.__reward__(new_state, action)
        self.current_state = new_state
        self.t += self.dt
        return new_state, reward, terminal_state, {}
    
    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.__successor_state__(state, action)
            if successor_state == state: continue
            ret.append((action, successor_state))
        return ret
    
    def is_goal(self, state):
        L,A,G,T,P,K,S = state.state
        # Attention that we do not break up to early since even at large W it is still possible that A_PB is violated!
        return self.A_PB - A > 0 and self.direct_W(L, G, P, K, S) > 2.e6 and P > 1e10 and self.t>400
    
    def is_terminal(self, state):
        L,A,G,T,P,K,S = state.state
        Leff=max(L-self.simulation_parameters['L0'], 0) if state.action[-1] else L
        W=self.direct_W(Leff, G, P, K, S)
        return not (A > self.A_PB or W < self.W_PB or P<self.P_PB)
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            ret_states_trace.append(self.__successor_state__(state, action))
        return ret_states_trace


"""
This is the implementation of the c:GLOBAL Environment in the form
that it can used within the Agent-Environment interface 
in combination with the DRL-agent.

@author: Felix Strnad
"""