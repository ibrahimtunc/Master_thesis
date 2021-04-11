# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:00:04 2021

@author: Ibrahim Alperen Tunc
"""
import matplotlib.pyplot as plt
import numpy as np
import wef_helper_functions as hlp
from cycler import cycler

dt = 0.00005 #time step for the model in seconds
tdur = 2 #duration of the stimulus in seconds.
t = np.arange(0,tdur, dt)
#test the model
stimulus = np.zeros(t.shape)
bspiketimes = hlp.LIF_reduced(stimulus, deltat=dt) #baseline activity

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
npersegRAM = 2**15 #I might try different values in the future for this.
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
npersegSAM = np.round(2**(15+np.log2(dt*contrastf/2))) * 1/(dt*contrastf) * 2


#Start with the systematic check with different coherences and frequencies
#time and remaining variables
dt = 0.00005 #time step for the model in seconds
tdur = 10 #duration of the stimulus in seconds.
t = np.arange(0, tdur, dt)
npersegRAM = 2**15 #I might try different values in the future for this.

#Kernel parameters and kernel generation
kernelparams = {'sigma' : 0.0001, 'lenfactor' : 5, 'resolution' : dt} #sigma will change in the next step 
kernel, kerneltime = hlp.spike_gauss_kernel(**kernelparams)

#Contrast and frequency variables
fAMs = np.logspace(np.log10(1),np.log10(300),101)
contrasts = np.linspace(0.001,0.1,100)

#preallocate the arrays
rpsSAM = np.zeros([len(fAMs),len(contrasts)]) #SAM response powers
rpsRAM = [] #RAM response powers
spsSAM = np.zeros([len(fAMs),len(contrasts)]) #SAM stimulus powers
spsRAM = [] #RAM stimulus powers
tfsRAM = [] #RAM transfer functions
fssSAM = [] #SAM stimulus/response frequency values (these will be different for different iterations since nperseg 
            #shows slight variations for different AM frequencies.). This is along the dimension of fAMs, not contrasts!
            
#gRAM = [] #stimulus-response coherence gamma for RAM, leave it for now as the system is noiseless
RAMwave = hlp.whitenoise(cflow, cfup, dt, tdur, rng=np.random) #use the same RAM stimulus with different contrasts

for cidx, contrast in enumerate(contrasts):
    #RAM stimulus
    RAMstim = contrast * RAMwave
    RAMspkt = hlp.LIF_reduced(RAMstim) #RAM spike times
    RAMio = hlp.power_spectra_transfer_funcs(RAMstim, RAMspkt, t[1:], kernel, npersegRAM)
    prRAM, psRAM, frRAM, fsRAM = RAMio.power_spectrum()
    _, csdRAM, fgamma, gamma = RAMio.cross_spectral_density(calcoherence=True) #in noiseless case coherence is (apparently) always 1!
    rpsRAM.append(prRAM[(frRAM<300) & (frRAM>0)])
    spsRAM.append(psRAM[(frRAM<300) & (frRAM>0)])
    tfsRAM.append(np.abs(csdRAM/psRAM)[(frRAM<300) & (frRAM>0)]) #the argument inside is the same as np.sqrt(prRAM/psRAM) (checked numerically)
    
    #SAM stimulus
    for fidx, freq in enumerate(fAMs):
        SAMstim = contrast*np.sin(2*np.pi*freq*t)
        SAMspkt = hlp.LIF_reduced(SAMstim) #SAM spike times
        npersegSAM = np.round(2**(15+np.log2(dt*freq/2))) * 1/(dt*freq) * 2
        SAMio = hlp.power_spectra_transfer_funcs(SAMstim, SAMspkt, t, kernel, npersegSAM)
        prSAM, psSAM, frSAM, fsSAM = SAMio.power_spectrum()
        psiSAM, priSAM = SAMio.interpolate_power_spectrum_value(freq, fsSAM, psSAM, frSAM, prSAM)
        rpsSAM[fidx,cidx] = priSAM
        spsSAM[fidx,cidx] = psiSAM 
    print(cidx)
tfsSAM = np.sqrt(rpsSAM/spsSAM)


#plottings
#general parameters.
SAMcols = plt.cm.Reds(np.linspace(0.2,1,4))  
RAMcols = plt.cm.Blues(np.linspace(0.2,1,4))  
subplots_adjust = {'top' : 0.882,
                   'bottom' : 0.061,
                   'left' : 0.082,
                   'right' : 0.992,
                   'hspace' : 0.725,
                   'wspace' : 0.06}

#1) Transfer functions for different contrasts
#1.1 RAM
fig, axs = plt.subplots(5,5, sharex=True, sharey='row')
axs[0,2].set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
axs[2,4].set_xlabel('Frequency [Hz]')
axs = axs.flatten()
plotnum = 0
for ax in axs:
    pnum = 0
    ax.set_prop_cycle(cycler('color', RAMcols))
    ax.set_title('[%.3f, %.3f]' %(contrasts[plotnum],contrasts[plotnum+3]))
    ax.set_xticks(np.linspace(0,300,15))
    while pnum <= 3:
        ax.plot(fsRAM[(frRAM<300) & (frRAM>0)], tfsRAM[plotnum])
        pnum += 1
        plotnum +=1
    #ax.legend()
fig.suptitle('RAM stimulus transfer functions for different contrasts')
plt.subplots_adjust(**subplots_adjust)

#1.2 SAM
fig, axs = plt.subplots(5,5, sharex=True, sharey='row')
axs = axs.flatten()
plotnum = 0
for ax in axs:
    pnum = 0
    ax.set_prop_cycle(cycler('color', SAMcols))
    ax.set_title('[%.3f, %.3f]' %(contrasts[plotnum],contrasts[plotnum+3]))
    ax.set_xticks(np.linspace(0,300,15))
    while pnum <= 3:
        ax.plot(fAMs, tfsSAM[:, plotnum])
        pnum += 1
        plotnum +=1
    #ax.legend()
fig.suptitle('SAM stimulus transfer functions for different contrasts')
plt.subplots_adjust(**subplots_adjust)

#2) Response powers as a function of contrast for different frequencies
#2.1 SAM
fig, axs = plt.subplots(5,5, sharex=True, sharey='row')
axs[0,2].set_ylabel('Power ' r'[$\frac{Hz**2}{Hz}$]')
axs[2,4].set_xlabel('Contrast [%]')
axs = axs.flatten()
plotnum = 0
for ax in axs:
    pnum = 0
    ax.set_prop_cycle(cycler('color', SAMcols))
    ax.set_title('[%.2f, %.2f]' %(fAMs[plotnum],fAMs[plotnum+3]))
    ax.set_xticks(np.linspace(0,300,15))
    while pnum <= 3:
        ax.plot(contrasts, tfsSAM[plotnum,:])
        pnum += 1
        plotnum +=1
fig.suptitle('SAM stimulus response powers for different frequencies')
plt.subplots_adjust(**subplots_adjust)

#2.2 RAM
fig, axs = plt.subplots(5,5, sharex=True, sharey='row')
axs[0,2].set_ylabel('Power ' r'[$\frac{Hz**2}{Hz}$]')
axs[2,4].set_xlabel('Contrast [%]')
axs = axs.flatten()
plotnum = 0
RAMfreqs = frRAM[(frRAM<300) & (frRAM>0)][::5] #99 frequency values here, so last plot will have 3 curves
for ax in axs:
    pnum = 0
    ax.set_prop_cycle(cycler('color', RAMcols))
    ax.set_xticks(np.linspace(0,300,15))
    if plotnum+3 < 98:
        ax.set_title('[%.2f, %.2f]' %(fAMs[plotnum],fAMs[plotnum+3]))
    else:
        ax.set_title('[%.2f, %.2f]' %(fAMs[plotnum],fAMs[plotnum+2]))
        pnum +=1 #this is a cheap method to plot only 3 curves in the last subplot
    while pnum <= 3:
        ax.plot(contrasts, tfsSAM[plotnum,:])
        pnum += 1
        plotnum +=1
fig.suptitle('RAM stimulus response powers for different frequencies')
plt.subplots_adjust(**subplots_adjust)
