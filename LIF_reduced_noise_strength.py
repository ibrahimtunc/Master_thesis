# -*- coding: utf-8 -*-
"""
Created on Sun May  9 09:38:47 2021

@author: Ibrahim Alperen Tunc
"""

import matplotlib.pyplot as plt
import numpy as np
import wef_helper_functions as hlp

#Test the reduced LIF model with regards to noise -> check how noise level affects coefficient of variation
#we want to have a noise level which leads to a CV between 0.3-0.5

#model parameters
noise_strengths = np.logspace(np.log10(0.00001), np.log10(0.5), 1001) #this will be an array to see what happens
dt = 0.00005 #time step for the model in seconds, seems not to interfere with cv much.
#prepare stimulus
tdur = 5 #stimulus duration in seconds
t = np.arange(0,tdur, dt)
stimulus = np.zeros(len(t))
#simulation parameters
ntrials = 200 #number of repetititons for each noise strength to average over

#preallocate arrays
cvs = np.zeros([ntrials, len(noise_strengths)]) #coefficient of variation
bfrs = np.zeros([ntrials, len(noise_strengths)]) #baseline firing rate

for idx, ns in enumerate(noise_strengths):
    for nidx in range(ntrials):    
        spkt = hlp.LIF_reduced(stimulus, deltat=dt, noise_strength=ns) #baseline activity
        spkt = spkt[spkt>0.1] #discard first 100 ms.
        cv = hlp.coefficient_of_variation(spkt)
        bfr = len(spkt) / tdur
        cvs[nidx, idx] = cv
        bfrs[nidx, idx] = bfr
    print('%.1f %% is finished.' %((100*idx/len(noise_strengths))+1))

#average values over trials
cvsavg = np.mean(cvs, axis=0)
bfrsavg = np.mean(bfrs, axis=0)


#plot noise-cv and noise-bfr
hlp.plotter.misc.plot_cv_bfr(noise_strengths, cvsavg, bfrsavg)
