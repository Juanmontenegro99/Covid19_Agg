import numpy as np
import math

def pomp_transition(var, rate, dt=1, num_ensembles=200):
    kb = np.maximum(1.0 - np.exp(-rate*dt), 0)
    num_ind   = np.random.binomial(list(var), kb )
    return np.squeeze(num_ind)

def modelH(x, beta, ihr, hfr, alpha, N, num_ensembles=200):

    kappa   = 1/6.4 # Incubation Period [days]
    gamma   = 1/4   # Recovery Period   [days]
    sigma   = 0.5   # Relative Unreported tranmissibility  [Adimensional]
    delta   = 1/365 # Reinfection period [days]

    S   = x[0,:]   # Susceptibles
    E   = x[1,:]   # Exposed
    Ir  = x[2,:]   # Infected Reported
    Iu  = x[3,:]   # Infected Un-reported
    Ih  = x[4,:]
    R   = x[5,:]   # Recovered
    H   = x[6,:]
    C   = x[7,:]   # Incident Cases
    D   = x[8,:]   # Incident Deaths

    foi =  beta * (Ir + sigma*Iu) / N

    # Stochastic transitions
    s2e      =  pomp_transition(S,  foi, num_ensembles=num_ensembles)                 # susceptible to exposed
    e2iu     =  pomp_transition(E,  (1-alpha)*kappa, num_ensembles=num_ensembles)     # exposed to infected underreported
    e2ir     =  pomp_transition(E,  alpha*(1-ihr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die
    e2ih     =  pomp_transition(E,  ihr*kappa, num_ensembles=num_ensembles)           # exposed to infected hospitalized
    iu2r     =  pomp_transition(Iu, gamma, num_ensembles=num_ensembles)              # infected under-reported to recovered
    ir2r     =  pomp_transition(Ir, gamma, num_ensembles=num_ensembles)              # infected reported to recovered
    ih2r     =  pomp_transition(Ih, (1-hfr)*1/10, num_ensembles=num_ensembles)            # infected hospitalized (who are not going to die) to recovered
    ih2death =  pomp_transition(Ih, hfr*1/10, num_ensembles=num_ensembles)                # infected hospitalized (who are going to die) to Death
    r2s      =  pomp_transition(R, delta, num_ensembles=num_ensembles)

    # Updates
    S  = S  - s2e + r2s                   # Susceptible
    E  = E  + s2e  - e2ir - e2iu - e2ih   # Exposed
    Ir = Ir + e2ir - ir2r                 # Infected reported
    Iu = Iu + e2iu - iu2r                 # Infected un-reported
    Ih = Ih + e2ih  - ih2death - ih2r     # Infected hospitalized
    R  = R  + ir2r + iu2r + ih2r - r2s    # Recovered
    H  = e2ih                             # Hospitalized
    C  = e2ir + e2ih                      # Incident Cases
    D  = ih2death                         # Incident Deaths

    return np.array([S, E, Ir, Iu, Ih, R, H, C, D])

def init_modelH(pop, num_ensembles=300, num_variables=8):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 3
        Ir0  = 1
        Iu0  = 1
        S0   = S0-E0-Ir0-Iu0
        Ih0  = 0
        R0   = 0
        H0   = 0
        C0   = Ir0
        D0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Ih0, R0, H0, C0, D0]

    return x_init

def model(x, beta, ifr, alpha, N, num_ensembles=200):

    kappa   = 1/6.4 # Incubation Period [days]
    gamma   = 1/4   # Recovery Period   [days]
    sigma   = 0.5   # Relative Unreported tranmissibility [Adimensional]
    delta   = 1/365 # Reinfection period [days]
    gamma_d = 1/12  # Death Period [days]

    S   = x[0,:]   # Susceptibles
    E   = x[1,:]   # Exposed
    Ir  = x[2,:]   # Infected Reported
    Iu  = x[3,:]   # Infected Un-reported
    Id  = x[4,:]
    R   = x[5,:]   # Recovered
    C   = x[6,:]   # Incident Cases
    D   = x[7,:]   # Incident Deaths

    foi =  beta * (Ir + sigma*Iu) / N

    # Stochastic transitions
    s2e      = pomp_transition(S, foi, num_ensembles=num_ensembles)                 # susceptible to exposed
    e2iu     = pomp_transition(E, (1-alpha)*kappa, num_ensembles=num_ensembles)     # exposed to infected underreported
    e2ir     = pomp_transition(E, alpha*(1-ifr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die
    e2id     = pomp_transition(E, ifr*kappa, num_ensembles=num_ensembles)           # exposed to infected hospitalized
    iu2r     = pomp_transition(Iu, gamma, num_ensembles=num_ensembles)              # infected under-reported to recovered
    ir2r     = pomp_transition(Ir, gamma, num_ensembles=num_ensembles)              # infected reported to recoveredto recovered
    id2death = pomp_transition(Id, gamma_d, num_ensembles=num_ensembles)            # infected hospitalized (who are going to die) to Death
    r2s      = pomp_transition(R, delta, num_ensembles=num_ensembles)

    # Updates
    S  = S - s2e + r2s                 # Susceptible
    E  = E + s2e  - e2ir - e2iu - e2id # Exposed
    Ir = Ir + e2ir - ir2r              # Infected reported
    Iu = Iu + e2iu - iu2r              # Infected un-reported
    Id = Id + e2id  - id2death         # Infected hospitalized
    R  = R + ir2r + iu2r - r2s         # Recovered
    C  = e2ir + e2id                   # Incident Cases
    D  = id2death                      # Incident Deaths
    print(- s2e + r2s + s2e - e2ir - e2iu - e2id + e2ir - ir2r + e2iu - iu2r + e2id + ir2r + iu2r - r2s)

    return np.array([S, E, Ir, Iu, Id, R, C, D])

def init_model(pop, num_ensembles=300, num_variables=8):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 3
        Ir0  = 1
        Iu0  = 1
        S0   = S0-E0-Ir0-Iu0
        Id0  = 0
        R0   = 0
        C0   = Ir0
        D0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Id0, R0, C0, D0]

    return x_init

def susceptible2exposed(susceptible_a, foi_a, dt=1):
    kb = np.maximum(1.0 - math.exp(-foi_a*dt), 0)
    return np.random.binomial(susceptible_a, kb )


def modelVH(x, beta, ihr, hfr, iVhr, hVfr, Vr, alpha, epsilon, N, num_ensembles=200):

    # Incubation Period
    kappa   = 1/6.4 #days

    # Recovery Period
    gamma   = 1/4 #days

    # Death Period
    gamma_d = 1/12 #days

    # Relative Unreported tranmissibility
    sigma   = 0.5 # Adimensional
    
    # One doses vaccine efficacy in reducing infectivity
    VEI = (0.762)
    
    delta = 1/182

    S   = x[0,:]
    E   = x[1,:]   # List with Exposed individuals in all age groups
    Ir  = x[2,:]   # List with reported Infected Individuals in all age groups
    Iu  = x[3,:]   # List with Un-reported Infected individuals in all age groups
    Ih  = x[4,:]   # List with Infected individuals who eventually will die (We)
    R   = x[5,:]   # List with Recovered and transient inmmune individuals
    H   = x[6,:]
    C   = x[7,:]   # List with incident cases in all age groups
    D   = x[8,:]   # List with incident Deaths in all age groups

    SV  = x[9,:]
    EV  = x[10,:]     # List with Exposed individuals with one vaccine dose in all age groups
    IVr = x[11,:]   # List with reported Infected Individuals with one vaccine dose in all age groups
    IVu = x[12,:]   # List with Un-reported Infected individuals with one vaccine dose in all age groups
    IVh = x[13,:]   # List with Infected individuals with one vaccine dose who eventually will die (We)
    RV  = x[14,:]   # List with Recovered and transient inmmune individuals with one vaccine dose
    HV  = x[15,:]
    CV  = x[16,:]   # List with incident cases with one vaccine dose in all age groups
    DV  = x[17,:]   # List with incident Deaths with one vaccine dose in all age groups



    # Compute force of infection in each age group FOI_a | Just proportional to the number of infected reported individuals and the number of infected under-reported individuals
    foi = beta * (((Ir + sigma*Iu)+((1-VEI)*(IVr + sigma*IVu)))/ N)    
        
    ###### TRANSIIONS ######
    
    s2e      =  pomp_transition(S,  foi, num_ensembles=num_ensembles)                 # susceptible to exposed
    s2sV    =  pomp_transition(S,  Vr, num_ensembles=num_ensembles)
    e2eV    =  pomp_transition(E,  Vr, num_ensembles=num_ensembles)
    e2iu     =  pomp_transition(E,  (1-alpha)*kappa, num_ensembles=num_ensembles)     # exposed to infected underreported
    e2ir     =  pomp_transition(E,  alpha*(1-ihr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die die
    e2ih     =  pomp_transition(E,  (1-epsilon)*ihr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iu2r     =  pomp_transition(Iu, gamma, num_ensembles=num_ensembles)               # infected under-reported to recovered
    ir2r     =  pomp_transition(Ir, gamma, num_ensembles=num_ensembles)               # infected reported (who are not going to die) to recovered
    r2rV    =  pomp_transition(R,  Vr, num_ensembles=num_ensembles)
    ih2r     =  pomp_transition(Ih, (1-hfr)*1/12, num_ensembles=num_ensembles)
    ih2death =  pomp_transition(Ih, hfr*1/12, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    r2s      =  pomp_transition(R, delta, num_ensembles=num_ensembles)
    
    sV2eV      =  pomp_transition(SV,  foi, num_ensembles=num_ensembles)                   # susceptible to exposed
    eV2iVu     =  pomp_transition(EV,  (1-alpha)*kappa, num_ensembles=num_ensembles)       # exposed to infected underreported
    eV2iVr     =  pomp_transition(EV,  alpha*(1-iVhr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die die
    eV2iVh     =  pomp_transition(EV,  (1-epsilon)*iVhr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iVu2rV     =  pomp_transition(IVu, gamma, num_ensembles=num_ensembles)                 # infected under-reported to recovered
    iVr2rV     =  pomp_transition(IVr, gamma, num_ensembles=num_ensembles)                 # infected reported (who are not going to die) to recovered
    iVh2rV     =  pomp_transition(IVh, (1-hVfr)*1/5, num_ensembles=num_ensembles)
    iVh2deathV =  pomp_transition(IVh, hVfr*1/5, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    rV2s        =  pomp_transition(RV, delta, num_ensembles=num_ensembles)
    
    S  = S    - s2e  - s2sV + r2s + rV2s         # Susceptible
    E  = E    + s2e  - e2ir - e2iu - e2ih - e2eV # Exposed
    Ir = Ir   + e2ir - ir2r                      # Infected reported
    Iu = Iu   + e2iu - iu2r                      # Infected un-reported
    Ih = Ih   + e2ih  - ih2death - ih2r          # Infected who are going to die
    R  = R    + ir2r + iu2r - r2rV  +ih2r - r2s  # Recovered
    H  = e2ih
    C  = e2ir + e2ih                             # Incident Cases
    D  = ih2death                                # Incident Deaths

    SV  = SV  - sV2eV  + s2sV                            # Susceptible
    EV  = EV  + sV2eV  - eV2iVr - eV2iVu - eV2iVh + e2eV # Exposed
    IVr = IVr + eV2iVr - iVr2rV                          # Infected reported
    IVu = IVu + eV2iVu - iVu2rV                          # Infected un-reported
    IVh = IVh + eV2iVh - iVh2deathV - iVh2rV             # Infected who are going to die
    RV  = RV  + iVr2rV + iVu2rV + r2rV + iVh2rV - rV2s   # Recovered
    HV  = eV2iVh                              
    CV  = eV2iVr + eV2iVh                                # Incident Cases
    DV  = iVh2deathV                                     # Incident Deaths
  

    return np.array([S, E, Ir, Iu, Ih, R, H, C, D, SV, EV, IVr, IVu, IVh, RV, HV, CV, DV])
#             S, E, Ir ,Iu ,Ih ,R ,H ,C ,D ,SV ,EV ,IVr ,IVu ,IVh ,RV ,HV ,CV ,DV

def init_modelVH(pop, num_ensembles=300, num_variables=8, num_age_groups=16):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 100204#12588
        Ir0  = 20618#4196
        Iu0  = 62215#4196
        Ih0  = 4322#244
        R0   = 1.903248e+07#2092063
        SV0   = 6310
        EV0   = 0
        IVr0  = 0
        IVu0  = 0
        S0   = 7.955899e+06#S0-E0-Ir0-Iu0-SV0-EV0-IVr0-IVu0-Ih0-R0
        
        H0   = Ih0
        C0   = Ir0
        D0   = 229
        
        IVh0  = 0
        RV0   = 0
        HV0   = 0
        CV0   = IVr0
        DV0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Ih0, R0, H0, C0, D0, SV0, EV0, IVr0, IVu0, IVh0, RV0, HV0, CV0, DV0]

    return x_init

def modelV(x, beta, ifr, iVfr, Vr, alpha, N, num_ensembles=200):

    
    kappa   = 1/6.4 # Incubation Period [days]
    gamma   = 1/4   # Recovery Period [days]
    gamma_d = 1/12  # Death Period [days]
    sigma   = 0.5   # Relative Unreported tranmissibility [Adimensional]
    VEI = (0.762)   # One doses vaccine efficacy in reducing infectivity [Adimensional]
    
    delta = 1/182

    S   = x[0,:]
    E   = x[1,:] # List with Exposed individuals in all age groups
    Ir  = x[2,:] # List with reported Infected Individuals in all age groups
    Iu  = x[3,:] # List with Un-reported Infected individuals in all age groups
    Id  = x[4,:] # List with Infected individuals who eventually will die (We)
    R   = x[5,:] # List with Recovered and transient inmmune individuals
    C   = x[6,:] # List with incident cases in all age groups
    D   = x[7,:] # List with incident Deaths in all age groups

    SV  = x[8,:]
    EV  = x[9,:]  # List with Exposed individuals with one vaccine dose in all age groups
    IVr = x[10,:] # List with reported Infected Individuals with one vaccine dose in all age groups
    IVu = x[11,:] # List with Un-reported Infected individuals with one vaccine dose in all age groups
    IVd = x[12,:] # List with Infected individuals with one vaccine dose who eventually will die (We)
    RV  = x[13,:] # List with Recovered and transient inmmune individuals with one vaccine dose
    CV  = x[14,:] # List with incident cases with one vaccine dose in all age groups
    DV  = x[15,:] # List with incident Deaths with one vaccine dose in all age groups



    # Compute force of infection in each age group FOI_a | Just proportional to the number of infected reported individuals and the number of infected under-reported individuals
    foi = beta * (((Ir + sigma*Iu)+((1-VEI)*(IVr + sigma*IVu)))/ N)    
        
    ###### TRANSIIONS ######
    
    s2e      = pomp_transition(S, foi, num_ensembles=num_ensembles)                 # susceptible to exposed
    s2sV     = pomp_transition(S, Vr, num_ensembles=num_ensembles)
    e2eV     = pomp_transition(E, Vr, num_ensembles=num_ensembles)
    e2iu     = pomp_transition(E, (1-alpha)*kappa, num_ensembles=num_ensembles)     # exposed to infected underreported
    e2ir     = pomp_transition(E, alpha*(1-ifr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die
    e2id     = pomp_transition(E, ifr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iu2r     = pomp_transition(Iu, gamma, num_ensembles=num_ensembles)              # infected under-reported to recovered
    ir2r     = pomp_transition(Ir, gamma, num_ensembles=num_ensembles)              # infected reported (who are not going to die) to recovered
    r2rV     = pomp_transition(R, Vr, num_ensembles=num_ensembles)
    id2death = pomp_transition(Id, gamma_d, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    r2s      = pomp_transition(R, delta, num_ensembles=num_ensembles)
    
    sV2eV      = pomp_transition(SV, foi, num_ensembles=num_ensembles)                  # susceptible to exposed
    eV2iVu     = pomp_transition(EV, (1-alpha)*kappa, num_ensembles=num_ensembles)      # exposed to infected underreported
    eV2iVr     = pomp_transition(EV, alpha*(1-iVfr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die
    eV2iVd     = pomp_transition(EV, iVfr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iVu2rV     = pomp_transition(IVu, gamma, num_ensembles=num_ensembles)               # infected under-reported to recovered
    iVr2rV     = pomp_transition(IVr, gamma, num_ensembles=num_ensembles)               # infected reported (who are not going to die) to recovered
    iVd2deathV = pomp_transition(IVd, gamma_d, num_ensembles=num_ensembles)             # infected reported (who are going to die) to Death
    rV2s       = pomp_transition(RV, delta, num_ensembles=num_ensembles)
    
    S  = S    - s2e  - s2sV + r2s + rV2s         # Susceptible
    E  = E    + s2e  - e2ir - e2iu - e2id - e2eV # Exposed
    Ir = Ir   + e2ir - ir2r                      # Infected reported
    Iu = Iu   + e2iu - iu2r                      # Infected un-reported
    Id = Id   + e2id - id2death                  # Infected who are going to die
    R  = R    + ir2r + iu2r - r2rV - r2s         # Recovered
    C  = e2ir + e2id                             # Incident Cases
    D  = id2death                                # Incident Deaths

    SV  = SV  - sV2eV  + s2sV                            # Susceptible
    EV  = EV  + sV2eV  - eV2iVr - eV2iVu - eV2iVd + e2eV # Exposed
    IVr = IVr + eV2iVr - iVr2rV                          # Infected reported
    IVu = IVu + eV2iVu - iVu2rV                          # Infected un-reported
    IVd = IVd + eV2iVd - iVd2deathV                      # Infected who are going to die
    RV  = RV  + iVr2rV + iVu2rV + r2rV - rV2s            # Recovered
    CV  = eV2iVr + eV2iVd                                # Incident Cases
    DV  = iVd2deathV                                     # Incident Deaths
  

    return np.array([S, E, Ir, Iu, Id, R, C, D, SV, EV, IVr, IVu, IVd, RV, CV, DV])

def init_modelV(pop, num_ensembles=300, num_variables=8, num_age_groups=16):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 102334#85763
        Ir0  = 21293#17036
        Iu0  = 63565#51214
        Id0  = 2621#2100
        R0   = 1.634106e+07#1.761464e+07#2092063 
        SV0   = 0
        EV0   = 0
        IVr0  = 0
        IVu0  = 0
        S0   = 7.314175e+06#S0-E0-Ir0-Iu0-SV0-EV0-IVr0-IVu0-Id0-R0
        
        C0   = Ir0
        D0   = 218#175
        
        IVd0  = 0
        RV0   = 0
        CV0   = IVr0
        DV0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Id0, R0, C0, D0, SV0, EV0, IVr0, IVu0, IVd0, RV0, CV0, DV0]

    return x_init