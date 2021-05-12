# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:09:45 2021

@author: Ibrahim Alperen Tunc
"""

import matplotlib.pyplot as plt
import numpy as np
import wef_helper_functions as hlp

#Simulate the model (noiseless, noisy) with sinus wave of different amplitudes and frequencies to see 
#how the firing rate changes with the characteristics of the sine wave.

#model parameters
noise_strengths = np.linspace(0, 0.05, 2) #one noiseless, one noisy case
dt = 0.00005 #time step for the model in seconds, seems not to interfere with cv much.

#prepare stimulus
ncycles = 3000 #number of cycles within the stimulus, set to 3000.
fAMs = np.linspace(1,300,300) #sine wave frequencies
contrasts = np.linspace(0.001,0.5,250) #sine wave amplitudes 

#Average firing rate. Shape len(noise_strengths) x len(fAMs) x len(contrasts)
frs = np.zeros([len(noise_strengths),len(fAMs), len(contrasts)]) 
#Average number of spikes (calculated cycle by cycle). Shape len(noise_strengths) x len(fAMs) x len(contrasts)
spkpercycs = np.zeros([len(noise_strengths), len(fAMs), len(contrasts)])

#run the simulations
for nidx, noise_strength in enumerate(noise_strengths):
    print('%i out of %i noise cases' %(nidx+1, len(noise_strengths)))
    
    """
    if noise_strength > 0:
        ncycles = ncycless * 10 #increase the number of cycles for the noisy model so that more data can be averaged
                                #to extract the signal out of noise
    else:
        ncycles = ncycless #in the noiseless case use less number of cycles for a faster computation
    """
    
    frs[nidx,:,:], spkpercycs[nidx,:,:] = hlp.model_fr_spk_per_cyc(ncycles, fAMs, contrasts, dt, noise_strength)
    
    
#plottings
titles = ['Average firing rate', 'Average spike # per cycle']
nrow = 5
ncol = 5

for nidx in range(len(noise_strengths)):
    #generate the plotter object for the current noise strength
    frplotter = hlp.plotter.fr_spkpercyc(frs[nidx,:,:], fAMs, contrasts, spkpercycs[nidx,:,:])
    
    #first figure: fAMs-contrasts color plots, color code shows avg firing rate (1st subplot) or avg number of 
    #spikes per cycle (2nd subplot)
    axs = frplotter.color_plot_rf_spkpercyc()
    
    #add the word 'noisy' to plot title in the noisy case
    if nidx > 0:
        noisetit = ' (noise %.2f)' %(noise_strength)
    else:
        noisetit = ''    
    axs[0].set_title(titles[0]+noisetit)
    axs[1].set_title(titles[1]+noisetit)
    #further figure aadjustments
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.5)
    plt.tight_layout()

    #plot r-contrast curves for different stimulus frequencies in separate subplots
    fig = frplotter.fr_contrast_subplots_for_f(nrow, ncol)    
    fig.suptitle('Average firing rate for different frequencies' + noisetit)
    #further figure aadjustments
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.5)
    plt.tight_layout()
    
    #plot r-f curves for different stimulus contrasts in separate subplots
    fig = frplotter.fr_f_subplots_for_contrasts(nrow,ncol)
    fig.suptitle('Average firing rate for different contrasts' + noisetit)
    #further figure aadjustments
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.5)
    plt.tight_layout()
    
    #plot avgspk-contrast curves for different stimulus frequencies in separate subplots
    fig = frplotter.avgspk_contrast_subplots_for_f(nrow,ncol)
    fig.suptitle('Average spike # for different frequencies' + noisetit)
    #further figure aadjustments
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.5)
    plt.tight_layout()
    
    #plot avgspk-f curves for different stimulus contrasts in separate subplots
    fig = frplotter.avgspk_f_subplots_for_contrasts(nrow,ncol)
    fig.suptitle('Average spike # for different contrasts'+noisetit)
    #further figure aadjustments
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.5)
    plt.tight_layout()
    