import numpy as np
import math

def pomp_transition(var, rate, dt=1, num_ensembles=200):
    kb = np.maximum(1.0 - np.exp(-rate*dt), 0)
    num_ind   = np.random.binomial(list(var), kb )
#     print(num_ind.shape)
    if num_ind.shape[-1]!=num_ensembles:
        print("Error transitioning stochastic model")
    return np.squeeze(num_ind)

def model(x, beta, ihr, hfr, alpha, N, num_ensembles=200):

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

def init_model(pop, num_ensembles=300, num_variables=8):

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

def susceptible2exposed(susceptible_a, foi_a, dt=1):
    kb = np.maximum(1.0 - math.exp(-foi_a*dt), 0)
    return np.random.binomial(susceptible_a, kb )


def modelV(x, beta, ihr, hfr, iV1hr, hV1fr, iV2hr, hV2fr, Vr1, Vr2, alpha, N, num_ensembles=200):

    # Incubation Period
    kappa   = 1/6.4 #days

    # Recovery Period
    gamma   = 1/4 #days

    # Death Period
    gamma_d = 1/12 #days

    # Relative Unreported tranmissibility
    sigma   = 0.5 # Adimensional
    
    # One doses vaccine efficacy in reducing infectivity
    V1EI = (0.673)
    
    # Two doses vaccine efficacy in reducing infectivity
    V2EI = (0.762)
    
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

    SV1  = x[9,:]
    EV1  = x[10,:]     # List with Exposed individuals with one vaccine dose in all age groups
    IV1r = x[11,:]   # List with reported Infected Individuals with one vaccine dose in all age groups
    IV1u = x[12,:]   # List with Un-reported Infected individuals with one vaccine dose in all age groups
    IV1h = x[13,:]   # List with Infected individuals with one vaccine dose who eventually will die (We)
    RV1  = x[14,:]   # List with Recovered and transient inmmune individuals with one vaccine dose
    HV1  = x[15,:]
    CV1  = x[16,:]   # List with incident cases with one vaccine dose in all age groups
    DV1  = x[17,:]   # List with incident Deaths with one vaccine dose in all age groups

    SV2  = x[18,:]
    EV2  = x[19,:]     # List with Exposed individuals with one vaccine dose in all age groups
    IV2r = x[20,:]   # List with reported Infected Individuals with one vaccine dose in all age groups
    IV2u = x[21,:]   # List with Un-reported Infected individuals with one vaccine dose in all age groups
    IV2h = x[22,:]   # List with Infected individuals with one vaccine dose who eventually will die (We)
    RV2  = x[23,:]   # List with Recovered and transient inmmune individuals with one vaccine dose
    HV2  = x[24,:]
    CV2  = x[25,:]   # List with incident cases with one vaccine dose in all age groups
    DV2  = x[26,:]   # List with incident Deaths with one vaccine dose in all age groups



    # Compute force of infection in each age group FOI_a | Just proportional to the number of infected reported individuals and the number of infected under-reported individuals
    foi = beta * (((Ir + sigma*Iu)+((1-V1EI)*(IV1r + sigma*IV1u))+((1-V2EI)*(IV2r + sigma*IV2u)))/ N)    
        
    ###### TRANSIIONS ######
    
    s2e      =  pomp_transition(S,  foi, num_ensembles=num_ensembles)                 # susceptible to exposed
    s2sV1    =  pomp_transition(S,  Vr1, num_ensembles=num_ensembles)
    e2eV1    =  pomp_transition(E,  Vr1, num_ensembles=num_ensembles)
    e2iu     =  pomp_transition(E,  (1-alpha)*kappa, num_ensembles=num_ensembles)     # exposed to infected underreported
    e2ir     =  pomp_transition(E,  alpha*(1-ihr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die die
    e2ih     =  pomp_transition(E,  ihr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iu2r     =  pomp_transition(Iu, gamma, num_ensembles=num_ensembles)               # infected under-reported to recovered
    ir2r     =  pomp_transition(Ir, gamma, num_ensembles=num_ensembles)               # infected reported (who are not going to die) to recovered
    r2rV1    =  pomp_transition(R,  Vr1, num_ensembles=num_ensembles)
    ih2r     =  pomp_transition(Ih, (1-hfr)*1/13, num_ensembles=num_ensembles)
    ih2death =  pomp_transition(Ih, hfr*1/13, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    r2s      =  pomp_transition(R, delta, num_ensembles=num_ensembles)
    
    sV12eV1      =  pomp_transition(SV1,  foi, num_ensembles=num_ensembles)                   # susceptible to exposed
    sV12sV2      =  pomp_transition(SV1,  Vr2, num_ensembles=num_ensembles)
    eV12eV2      =  pomp_transition(EV1,  Vr2, num_ensembles=num_ensembles)
    eV12iV1u     =  pomp_transition(EV1,  (1-alpha)*kappa, num_ensembles=num_ensembles)       # exposed to infected underreported
    eV12iV1r     =  pomp_transition(EV1,  alpha*(1-iV1hr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die die
    eV12iV1h     =  pomp_transition(EV1,  iV1hr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iV1u2rV1     =  pomp_transition(IV1u, gamma, num_ensembles=num_ensembles)                 # infected under-reported to recovered
    iV1r2rV1     =  pomp_transition(IV1r, gamma, num_ensembles=num_ensembles)                 # infected reported (who are not going to die) to recovered
    rV12rV2      =  pomp_transition(RV1,  Vr2, num_ensembles=num_ensembles)
    iV1h2rV1     =  pomp_transition(IV1h, (1-hV1fr)*1/5, num_ensembles=num_ensembles)
    iV1h2deathV1 =  pomp_transition(IV1h, hV1fr*1/5, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    rV12s        =  pomp_transition(RV1, delta, num_ensembles=num_ensembles)
    
    sV22eV2      =  pomp_transition(SV2,  foi, num_ensembles=num_ensembles)                   # susceptible to exposed
    eV22iV2u     =  pomp_transition(EV2,  (1-alpha)*kappa, num_ensembles=num_ensembles)       # exposed to infected underreported
    eV22iV2r     =  pomp_transition(EV2,  alpha*(1-iV2hr)*kappa, num_ensembles=num_ensembles) # exposed to infected reported who are not going to die die
    eV22iV2h     =  pomp_transition(EV2,  iV2hr*kappa, num_ensembles=num_ensembles)           # exposed to infected reported who are going to die
    iV2u2rV2     =  pomp_transition(IV2u, gamma, num_ensembles=num_ensembles)                 # infected under-reported to recovered
    iV2r2rV2     =  pomp_transition(IV2r, gamma, num_ensembles=num_ensembles)                 # infected reported (who are not going to die) to recovered
    iV2h2rV2     =  pomp_transition(IV2h, (1-hV2fr)*1/5, num_ensembles=num_ensembles)
    iV2h2deathV2 =  pomp_transition(IV2h, hV2fr*1/5, num_ensembles=num_ensembles)            # infected reported (who are going to die) to Death
    rV22s        =  pomp_transition(RV2, delta, num_ensembles=num_ensembles)
    
    S  = S    - s2e  - s2sV1 + r2s + rV12s + rV22s # Susceptible
    E  = E    + s2e  - e2ir - e2iu - e2ih - e2eV1  # Exposed
    Ir = Ir   + e2ir - ir2r                        # Infected reported
    Iu = Iu   + e2iu - iu2r                        # Infected un-reported
    Ih = Ih   + e2ih  - ih2death - ih2r            # Infected who are going to die
    R  = R    + ir2r + iu2r - r2rV1  +ih2r - r2s   # Recovered
    H  = e2ih
    C  = e2ir + e2ih                               # Incident Cases
    D  = ih2death                                  # Incident Deaths

    SV1  = SV1  - sV12eV1  + s2sV1 - sV12sV2                                  # Susceptible
    EV1  = EV1  + sV12eV1  - eV12iV1r - eV12iV1u - eV12iV1h + e2eV1 - eV12eV2 # Exposed
    IV1r = IV1r + eV12iV1r - iV1r2rV1                                         # Infected reported
    IV1u = IV1u + eV12iV1u - iV1u2rV1                                         # Infected un-reported
    IV1h = IV1h + eV12iV1h - iV1h2deathV1 - iV1h2rV1                          # Infected who are going to die
    RV1  = RV1  + iV1r2rV1 + iV1u2rV1 + r2rV1 + iV1h2rV1 - rV12s - rV12rV2    # Recovered
    HV1  = eV12iV1h                              
    CV1  = eV12iV1r + eV12iV1h                                                 # Incident Cases
    DV1  = iV1h2deathV1                                                       # Incident Deaths

    SV2  = SV2  - sV22eV2  + sV12sV2                                  # Susceptible
    EV2  = EV2  + sV22eV2  - eV22iV2r - eV22iV2u - eV22iV2h + eV12eV2 # Exposed
    IV2r = IV2r + eV22iV2r - iV2r2rV2                                 # Infected reported
    IV2u = IV2u + eV22iV2u - iV2u2rV2                                 # Infected un-reported
    IV2h = IV2h + eV22iV2h - iV2h2deathV2 - iV2h2rV2                  # Infected who are going to die
    RV2  = RV2  + iV2r2rV2 + iV2u2rV2 + rV12rV2 + iV2h2rV2 - rV22s    # Recovered
    HV2  = eV22iV2h                              
    CV2  = eV22iV2r + eV22iV2h                                         # Incident Cases
    DV2  = iV2h2deathV2                                               # Incident Deaths
    
#     print(- s2e  - s2sV + r2s + s2e  - e2ir - e2iu - e2ih - e2eV + e2ir - ir2r + e2iu - iu2r + e2ih  - ih2death - ih2r + ir2r + iu2r - r2rV  +ih2r - r2s - sV2eV  + s2sV + rV2sV + sV2eV  - eV2iVr - eV2iVu - eV2iVh + e2eV + eV2iVr - iVr2rV + eV2iVu - iVu2rV + eV2iVh - iVh2deathV - iVh2rV + iVr2rV + iVu2rV + r2rV + iVh2rV - rV2sV +ih2death +iVh2deathV)
  

    return np.array([S, E, Ir, Iu, Ih, R, H, C, D, SV1, EV1, IV1r, IV1u, IV1h, RV1, HV1, CV1, DV1, SV2, EV2, IV2r, IV2u, IV2h, RV2, HV2, CV2, DV2])
#             S, E, Ir ,Iu ,Ih ,R ,H ,C ,D ,SV ,EV ,IVr ,IVu ,IVh ,RV ,HV ,CV ,DV

def init_modelV(pop, num_ensembles=300, num_variables=8, num_age_groups=16):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 100950#12588
        Ir0  = 20472#4196
        Iu0  = 62502#4196
        Ih0  = 4309#244
        R0   = 2.534997e+07#2092063
        SV10   = 6310
        EV10   = 0
        IV1r0  = 0
        IV1u0  = 0
        S0   = S0-E0-Ir0-Iu0-SV10-EV10-IV1r0-IV1u0-Ih0-R0
        
        H0   = Ih0
        C0   = Ir0
        D0   = 229
        
        IV1h0  = 0
        RV10   = 0
        HV10   = 0
        CV10   = IV1r0
        DV10   = 0
        
        SV20   = 0
        EV20   = 0
        IV2r0  = 0
        IV2u0  = 0
        IV2h0  = 0
        RV20   = 0
        HV20   = 0
        CV20   = IV2r0
        DV20   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Ih0, R0, H0, C0, D0, SV10, EV10, IV1r0, IV1u0, IV1h0, RV10, HV10, CV10, DV10, SV20, EV20, IV2r0, IV2u0, IV2h0, RV20, HV20, CV20, DV20]

    return x_init