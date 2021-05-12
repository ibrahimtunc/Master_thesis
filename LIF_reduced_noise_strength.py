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
noise_strengths = np.linspace(0, 0.5, 101) #this will be an array to see what happens
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
        cv = hlp.coefficient_of_variation(spkt)
        bfr = len(spkt) / tdur
        cvs[nidx, idx] = cv
        bfrs[nidx, idx] = bfr
    print('%.1f %% is finished.' %((100*idx/len(noise_strengths))+1))

#average values over trials
cvsavg = np.mean(cvs, axis=0)
bfrsavg = np.mean(bfrs, axis=0)



#plot noise-cv and noise-bfr
fig, axs = plt.subplots(1,2, sharex=True)
axs[0].plot(noise_strengths, cvsavg, 'k.-')
axs[0].axvspan(noise_strengths[cvsavg>=0.3][0], noise_strengths[cvsavg<=0.5][-1], alpha=0.5, color='k')
axs[0].plot([np.min(noise_strengths),np.max(noise_strengths)], [0.3,0.3], 'k--')
axs[0].plot([np.min(noise_strengths),np.max(noise_strengths)], [0.5,0.5], 'k--')
axs[1].plot(noise_strengths, bfrsavg, 'k.-')
axs[0].set_xlabel('Noise strength')
axs[0].set_ylabel('CV')
axs[0].set_title('Coefficient of variation')
axs[1].set_ylabel('Firing rate [Hz]')
axs[1].set_xlabel('Noise strength')
axs[1].set_title('Average baseline firing rate')
plt.get_current_fig_manager().window.showMaximized()
plt.pause(0.5)
plt.tight_layout()
