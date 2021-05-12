# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:00:04 2021

@author: Ibrahim Alperen Tunc
"""
import matplotlib.pyplot as plt
import numpy as np
import wef_helper_functions as hlp
from cycler import cycler
import sys
import time

#SWITCH TO SCIPY CONVOLVE, NORMALIZE KERNEL TO UNIT AREA
test = False #test the model if true
if test == False:
    None
else:
    dt = 0.00005 #time step for the model in seconds
    tdur = 2 #duration of the stimulus in seconds.
    t = np.arange(0,tdur, dt)
    #test the model
    stimulus = np.zeros(t.shape)
    bspiketimes = hlp.LIF_reduced(stimulus, deltat=dt, noise_strength=0) #baseline activity
    bspiketimesn = hlp.LIF_reduced(stimulus, deltat=dt, noise_strength=0.05) #baseline activity with noise
    
    #check the spike times
    #fig, ax = plt.subplots(1,1)
    #ax.plot(spiketimes, np.ones(spiketimes.shape), '|', markersize=10, markeredgewidth=1.5)
    
    #SAM stimulus
    contrast = 0.05 #this will be an array in the future
    contrastf = 10 #this will be an array in the future
    SAMstim = contrast*np.sin(2*np.pi*contrastf*t)
    SAMspiketimes = hlp.LIF_reduced(SAMstim)
    #check the spike times
    #fig, axs = plt.subplots(1,2)
    #axs[0].plot(t, SAMstim, 'k-')
    #axs[1].plot(SAMspiketimes, np.ones(len(SAMspiketimes)), '|', markersize=10, markeredgewidth=1.5)
    
    #RAM stimulus
    cflow = 0
    cfup = 300 
    RAMstim = contrast * hlp.whitenoise(cflow, cfup, dt, tdur, rng=np.random) #its standard deviation is same as contrast
    RAMspiketimes = hlp.LIF_reduced(RAMstim)
    
    #Firing rates
    baselinefr = hlp.calculate_isi_frequency(bspiketimes, t) #baseline firing rate is 60 Hz
    SAMfr = hlp.calculate_isi_frequency(SAMspiketimes, t)
    RAMfr = hlp.calculate_isi_frequency(RAMspiketimes, t)
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    axs[0].plot(t[t>0.1], baselinefr[t>0.1], 'k-', label='Baseline')
    axs[1].plot(t[t>0.1], SAMfr[t>0.1], 'r-', label='SAM')
    axs[2].plot(t[t>0.1], RAMfr[t>0.1], 'b-', label='RAM')
    axs[0].set_xlim(0,tdur+0.1)
    axs[0].set_xticks(np.linspace(0,tdur,5))
    axs[1].set_xlabel('Time [s]')
    axs[0].set_ylabel('Firing rate [Hz]')
    fig.suptitle('Instantaneous firing rates with different stimuli')
    titles = ['Baseline', 'SAM', 'RAM']
    for idx, title in enumerate(titles):
        axs[idx].set_title(title)
    
    #convolve with a gaussian
    #sigma will be an array in the future to see how integration time window affects the stimulus-response relationships.
    kernelparams = {'sigma' : 0.001, 'lenfactor' : 5, 'resolution' : dt} 
    kernel, kerneltime = hlp.spike_gauss_kernel(**kernelparams)
    bconv, bspikes = hlp.convolved_spikes(bspiketimes, t, kernel) #baseline
    SAMconv, SAMspikes = hlp.convolved_spikes(SAMspiketimes, t, kernel) #SAM
    RAMconv, RAMspikes = hlp.convolved_spikes(RAMspiketimes, t, kernel) #RAM
    
    #stimulus/response power spectral densities:
    npersegRAM = 2**12 #I might try different values in the future for this.
    #For SAM, ensure that half of the segment with has integer number of wavelengths of given stimulus frequency. It is important to ensure
    #the half segment width, since in welch and csd the segments overlap by half of the data points. This way all segments include the same
    #integer number of wavelengths of the SAM stimulus.
    """
    Maths behinds nperseg rounding: => nperseg = nps
    nps/2 = q * wl #we want to have half nperseg as the multiple of SAM stimulus wavelength (wl), i.e. q waves should be inside nps/2
    q = nps/(2*wl)
    {wl = 1/f [s] = 1/(f*dt) [per time bin]} #convert the wavelength to per time bin as nperseg, f is stimulus frequency.
    q = nps*f*dt/2 = 2**(x+log2(f*dt/2)) #nps is written in power of 2, so to find the number of wavelengths fitting there, convert all
                                         #of the parameter values to powers of 2. x is the initial value we want to have (e.g. 15)
    #replace q to first equation:
    nps/2 = 2**(x+log2(f*dt/2)) * 1/(f*dt) <=> nps = 2**(x+log2(f*dt/2)) * 1/(f*dt) * 2
    #since we want to have integer number of waves per each segment, rounding the value q is crucial:
    nps = round(2**(x+log2(f*dt/2))) * 1/(f*dt) * 2    
    """
    npersegSAM = np.round(2**(12+np.log2(dt*contrastf/2))) * 1/(dt*contrastf) * 2

#Start with the systematic check with different coherences and frequencies

#time and remaining variables
dt = 0.00005 #time step for the model in seconds
tdur = 10 #duration of the stimulus in seconds.
t = np.arange(0, tdur, dt)
npersegex = 13 #exponent of nperseg
npersegRAM = 2**npersegex #I might try different values in the future for this.

#Kernel parameters and kernel generation
kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : dt} #sigma will change in the next step 
kernel, kerneltime = hlp.spike_gauss_kernel(**kernelparams)

#Contrasts 
contrasts = np.logspace(np.log10(0.001),np.log10(0.5),25) 

#baseline condition
stimulus = np.zeros(t.shape)
bline = hlp.simulation(1, stimulus, t, kernel, npersegRAM)

#RAM
#frequency cutoffs
cflow = 0
cfup = 300 
#number of repetitions (different white noise)
nrep = 10 
#preallocation
RAMios = np.zeros([nrep, len(contrasts)], dtype=object) #class object for power spectra
RAMfrs = np.zeros([nrep, len(contrasts)])

#simulation
for idx in range(nrep):
    #simulation for each white noise
    RAMwave = hlp.whitenoise(cflow, cfup, dt, tdur, rng=np.random) #use the same RAM stimulus with different contrasts
    
    #simulation for each contrast
    for cidx, contrast in enumerate(contrasts):
        RAMstim = contrast * RAMwave
        sobj = hlp.simulation(1, RAMstim, t[1:], kernel, npersegRAM) #simulation object
        RAMios[idx, cidx] = sobj.ios
        RAMfrs[idx,cidx] = sobj.frs
    print(idx)

#average firing rate
RAMfr = np.mean(RAMfrs, axis=0)

#calculate coherence and tf:
#preallocate
RAMtfs = []
RAMcoh = []
RAMpr = [] #average RAM response power
RAMps = [] #average RAM stimulus power
RAMprs = [] #average RAM csd
#iterate over each contrast to get the repetition trials
for idx in range(len(contrasts)):
    curioRAM = RAMios[:,idx] #all io objects with same contrast but different white noise
    #preallocate arrays
    RAMcsd = [] #cross spectral density
    RAMpss = [] #stimulus power
    RAMprr = [] #response power
    
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
    
#SAM
#frequency values
fAMs = np.linspace(1,300,3000) #this value will be something around thousands (6000 e.g.) 1000 is chosen due to time
                               #constraints entailing the simulation.

#SAM simulation is a little bit more tricky: you need to run too much simulations (for each fAM) but from each of
#those you need a few values (instead of arrays). For memory allocation optimization you need to plan accordingly.
#You know the shape of each array, namely len(fAMs) x len(contrasts), so you can accordingly implement it.

#preallocate
SAMpr = np.zeros([len(contrasts), len(fAMs)]) #average SAM response power
SAMps = np.zeros([len(contrasts), len(fAMs)]) #average SAM stimulus power
SAMprs = np.zeros([len(contrasts), len(fAMs)], dtype=complex) #average SAM csd
SAMfrs = np.zeros([len(contrasts), len(fAMs)]) #average SAM firing rates

for cidx, contrast in enumerate(contrasts):
    start = time.process_time()
    for fidx, fAM in enumerate(fAMs):
        npersegSAM = np.round(2**(npersegex+np.log2(dt*fAM/2))) * 1/(dt*fAM) * 2
        if npersegSAM == 0:
            npersegSAM = npersegRAM
        SAMstim = contrast*np.sin(2*np.pi*fAM*t)
        sobj = hlp.simulation(1, SAMstim, t, kernel, npersegSAM) #simulation object
        
        psi, pri, prsi = sobj.ios.interpolate_power_spectrum_value(fAM)
        SAMpr[cidx, fidx] = pri
        SAMps[cidx, fidx] = psi
        SAMprs[cidx, fidx] = prsi
        SAMfrs[cidx, fidx] = sobj.frs
    print(cidx, time.process_time()-start)
#average firing rate
SAMfr = np.mean(SAMfrs, axis=1)
#transfer function and coherence
SAMtfs = np.abs(SAMprs/SAMps) #this way is faster (in this case) than using the class object
SAMcoh = np.abs(SAMprs)**2 / (SAMps * SAMpr) 


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
frange = RAMios[0,0].fs[(RAMios[0,0].fs>=cflow) & (RAMios[0,0].fs<=cfup)]
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
fig = SAMplots.pr_plotter(SAMpr)
fig.suptitle('Response powers for SAM stimulus at different contrasts')

#2.2 csd
fig = SAMplots.csd_plotter(SAMprs)
fig.suptitle('SAM absolute cross spectral density at different contrasts')

#2.3 transfer function
fig = SAMplots.transfer_func_plotter(SAMtfs)
fig.suptitle('SAM stimulus transfer functions for different contrasts')

#2.4 s-r coherence
fig = SAMplots.s_r_coherence_plotter(SAMcoh)
fig.suptitle('SAM stimulus-response coherences for different contrasts')

#2.5 Response powers as a function of contrast for different frequencies
fig = SAMplots.pr_freq_plotter(SAMpr)
fig.suptitle('SAM stimulus response powers for different frequencies')
plt.pause(0.5)
plt.tight_layout()

