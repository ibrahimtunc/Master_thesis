# -*- coding: utf-8 -*-
"""
Created on Mon May  3 00:20:16 2021

@author: Ibrahim Alperen Tunc
"""

import matplotlib.pyplot as plt
import numpy as np
import wef_helper_functions as hlp
import time 
import sys

#Noisy version of the reduced LIF model

#model parameters
noise_strength = 0.05 #0.05 results in a coefficient of variation of 0.414.
dt = 0.00005 #time step for the model in seconds, seems not to interfere with cv much.

#time and remaining variables
tdur = 10 #duration of the stimulus in seconds.
t = np.arange(0, tdur, dt)
npersegex = 13 #exponent of nperseg
npersegRAM = 2**npersegex #I might try different values in the future for this.
nsimul = 10 #number of simulations per condition to average over (since model is noisy)

#Kernel parameters and kernel generation
kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : dt} #sigma will change in the next step 
kernel, kerneltime = hlp.spike_gauss_kernel(**kernelparams)

#Contrasts 
contrasts = np.logspace(np.log10(0.001),np.log10(0.5),25) 

#baseline condition
stimulus = np.zeros(t.shape)
bline = hlp.simulation(nsimul, stimulus, t, kernel, npersegRAM, noise_strength=noise_strength)
blinefravg = np.mean(bline.frs) #average baseline firing rate over 10 trials.

#RAM
#frequency cutoffs
cflow = 0
cfup = 300 
#number of repetitions (different white noise)
nrep = 10
#preallocation
RAMios = np.zeros([nsimul, nrep, len(contrasts)], dtype=object) #class object for power spectra
RAMfrs = np.zeros([nsimul, nrep, len(contrasts)])

#simulation
for idx in range(nrep):
    #simulation for each white noise
    RAMwave = hlp.whitenoise(cflow, cfup, dt, tdur, rng=np.random) #use the same RAM stimulus with different contrasts
    
    #simulation for each contrast
    for cidx, contrast in enumerate(contrasts):
        RAMstim = contrast * RAMwave
        sobj = hlp.simulation(nsimul, RAMstim, t[1:], kernel, npersegRAM, noise_strength=noise_strength) #simulation
        RAMios[:, idx, cidx] = sobj.ios
        RAMfrs[:, idx,cidx] = sobj.frs
    print(idx)

#average firing rate
RAMfr = np.mean(np.mean(RAMfrs, axis=1), axis=0) #first average over 2nd dimension (nrep, all simulations with same
                                                 #white noise and contrast) then over the first dimension 
                                                 #(nsimul, simulations with same contrast but different white noise)

#calculate coherence and tf:
#preallocate
RAMtfs = []
RAMcoh = []
RAMpr = [] #average RAM response power
RAMps = [] #average RAM stimulus power
RAMprs = [] #average RAM csd
#iterate over each contrast to get the repetition trials
for idx in range(len(contrasts)):
    #preallocate arrays
    RAMcsd = [] #cross spectral density
    RAMpss = [] #stimulus power
    RAMprr = [] #response power
    for sidx in range(nsimul):    
        curioRAM = RAMios[sidx, :, idx] #all io objects with same contrast but different white noise
        
        
        #iterate over each repetition trial (different white noise)
        for io in curioRAM: 
            RAMcsd.append(io.prs[(io.frs>=cflow) & (io.frs<=cfup)])
            RAMpss.append(io.ps[(io.fs>=cflow) & (io.fs<=cfup)])
            RAMprr.append(io.pr[(io.fr>=cflow) & (io.fr<=cfup)])
        
    #class object to calculate transfer function and coherence for the given contrast
    RAMfuncs = hlp.coherence_and_transfer_func(np.squeeze(RAMcsd), np.squeeze(RAMpss), np.squeeze(RAMprr))    
    #calculations
    RAMtfs.append(RAMfuncs.calculate_transfer_func())
    RAMcoh.append(RAMfuncs.calculate_sr_coherence())
    RAMpr.append(np.mean(RAMprr, axis=0))
    RAMps.append(np.mean(RAMpss, axis=0)) 
    RAMprs.append(np.mean(RAMcsd, axis=0))
RAMpr = np.squeeze(RAMpr)
RAMps = np.squeeze(RAMps)
RAMprs = np.squeeze(RAMprs)
RAMtfs = np.squeeze(RAMtfs)
RAMcoh = np.squeeze(RAMcoh)

#SAM
#frequency values
fAMs = np.linspace(1,300,500) #this value will be something around thousands (6000 e.g.) 500 is chosen due to time
                               #constraints entailing the simulation.

#SAM simulation is a little bit more tricky: you need to run too much simulations (for each fAM) but from each of
#those you need a few values (instead of arrays). For memory allocation optimization you need to plan accordingly.
#You know the shape of each array, namely len(fAMs) x len(contrasts), so you can accordingly implement it.

#preallocate
SAMpr = np.zeros([nsimul, len(contrasts), len(fAMs)]) #average SAM response power
SAMps = np.zeros([nsimul, len(contrasts), len(fAMs)]) #average SAM stimulus power
SAMprs = np.zeros([nsimul, len(contrasts), len(fAMs)], dtype=complex) #average SAM csd
SAMfrs = np.zeros([nsimul, len(contrasts), len(fAMs)]) #average SAM firing rates (time avg per trial)
SAMtfs = np.zeros([len(contrasts), len(fAMs)]) #average SAM firing rates
SAMcoh = np.zeros([len(contrasts), len(fAMs)]) #average SAM firing rates

for cidx, contrast in enumerate(contrasts):
    start = time.process_time()
    for fidx, fAM in enumerate(fAMs):
        npersegSAM = np.round(2**(npersegex+np.log2(dt*fAM/2))) * 1/(dt*fAM) * 2
        if npersegSAM == 0:
            npersegSAM = npersegRAM
        SAMstim = contrast*np.sin(2*np.pi*fAM*t)
        sobj = hlp.simulation(nsimul, SAMstim, t, kernel, npersegSAM, noise_strength=noise_strength) #simulation
        for oidx, obj in enumerate(sobj.ios):
            psi, pri, prsi = obj.interpolate_power_spectrum_value(fAM)
            SAMpr[oidx, cidx, fidx] = pri
            SAMps[oidx, cidx, fidx] = psi
            SAMprs[oidx, cidx, fidx] = prsi
        SAMfrs[:, cidx, fidx] = sobj.frs
    
    #transfer function and coherence
    SAMfunc = hlp.coherence_and_transfer_func(SAMprs[:,cidx,:], SAMps[:,cidx,:], SAMpr[:,cidx,:])
    SAMtfs[cidx, :] = SAMfunc.calculate_transfer_func() 
    SAMcoh[cidx, :] = SAMfunc.calculate_sr_coherence()
    print(cidx, (time.process_time()-start)/60)

#average firing rate
SAMfr = np.mean(np.mean(SAMfrs, axis=0), axis=-1)

#save session
directory = r'D:\ALPEREN\Tübingen NB\Semester 4\Thesis\git\codes\data'
flm = hlp.file_management(directory, globals(), dir())
flm.save_file('SAM_RAM_noisy_simul_12.05')


#plottings
#general parameters.
subplots_adjust = {'top' : 0.882,
                   'bottom' : 0.102,
                   'left' : 0.1,
                   'right' : 0.981,
                   'hspace' : 0.775,
                   'wspace' : 0.136}
nrows = 5
ncols = 5
#1) RAM
#frequency range
frange = RAMios[0,0,0].fs[(RAMios[0,0,0].fs>=cflow) & (RAMios[0,0,0].fs<=cfup)]
#plotter object
RAMplots = hlp.plotter.stimulus_response_plots(nrows, ncols, subplots_adjust, 
                                               frange, bline.frs, RAMfr, contrasts, SAM=False)

#1.1 Response power
fig = RAMplots.pr_plotter(RAMpr)
fig.suptitle('Response powers for RAM stimulus at different contrasts')

#1.2 csd 
fig = RAMplots.csd_plotter(RAMprs)
fig.suptitle('RAM absolute cross spectral density at different contrasts')

#1.3 Transfer functions for different contrasts
fig = RAMplots.transfer_func_plotter(RAMtfs)
fig.suptitle('RAM stimulus transfer functions for different contrasts')

#1.4 s-r coherence
fig = RAMplots.s_r_coherence_plotter(RAMcoh)
fig.suptitle('RAM stimulus-response coherences for different contrasts')

#1.5 Response powers as a function of contrast for different frequencies MOVE TO HLP
fig = RAMplots.pr_freq_plotter(RAMpr)
fig.suptitle('RAM stimulus response powers for different frequencies')
plt.pause(0.5)
plt.tight_layout()

#2 SAM

#plotter object
SAMplots = hlp.plotter.stimulus_response_plots(nrows, ncols, subplots_adjust, 
                                               fAMs, bline.frs, SAMfr, contrasts, SAM=True)

#2.1 Response power
fig = SAMplots.pr_plotter(np.mean(SAMpr, axis=0))
fig.suptitle('Response powers for SAM stimulus at different contrasts')

#2.2 csd
fig = SAMplots.csd_plotter(np.mean(SAMprs, axis=0))
fig.suptitle('SAM absolute cross spectral density at different contrasts')

#2.3 transfer function
fig = SAMplots.transfer_func_plotter(SAMtfs)
fig.suptitle('SAM stimulus transfer functions for different contrasts')

#2.4 s-r coherence
fig = SAMplots.s_r_coherence_plotter(SAMcoh)
fig.suptitle('SAM stimulus-response coherences for different contrasts')

#2.5 Response powers as a function of contrast for different frequencies
fig = SAMplots.pr_freq_plotter(np.mean(SAMpr, axis=0))
fig.suptitle('SAM stimulus response powers for different frequencies')
plt.pause(0.5)
plt.tight_layout()

