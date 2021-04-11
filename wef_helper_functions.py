# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:49:02 2021

@author: Ibrahim Alperen Tunc
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence
from scipy.interpolate import interp1d as interpolate
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit

figdict = {'axes.titlesize' : 30,
           'axes.labelsize' : 25,
           'xtick.labelsize' : 25,
           'ytick.labelsize' : 25,
           'legend.fontsize' : 25,
           'figure.titlesize' : 30,
           'image.cmap' : 'gray'}
plt.style.use(figdict)

#functions to be used in the master thesis

@jit(nopython=True)
def LIF_reduced(stimulus, v_zero=0.0, v_base=0.0, v_offset=1.5, mem_tau=0.015, threshold=1.0, deltat=0.00005):
    """ 
    Reduced leaky integrate and fire neuron. No adaptation, no refractoriness, no dendritic lowpass filtering
    and envelope extraction
    
    Parameters
    ----------
    stimulus: 1-D array
        Stimulus leading to model neuron spiking
    v_zero: float
        Initial membrane potential
    v_base: float
        Baseline membrane potential. When spike happens, membrane potential relaxes to this value
    v_offset: float
        Offset for the steady state membrane potential. In absence of stimulus, noise and threshold, the
        membrane potential would converge to this value over time. This value is crucial to obtain baseline
        firing rate.
    mem_tau: float
        Membrane time constant in seconds
    threshold: float
        Membrane potential threshold. When this value is exceeded, spike is recorded for that time and 
        membrane potential is set to the baseline value v_base.
    deltat: float
        Integration time in seconds. This value is used to digitize each steps of the differential equation
    
    Returns
    -------
    spike_times: 1-D array
        Simulated spike times in seconds.
    """  
    #LIFAC reduced to passive membrane linear LIF model -> No dendritic compartment, no refractory perid, no 
    #adaptation for now. Also dscard noise in the very first step.

    # initial conditions:
    v_mem = v_zero

    # rectify stimulus array:
    stimulus = stimulus.copy()
    stimulus[stimulus < 0.0] = 0.0

    # integrate:
    spike_times = []
    for i in range(len(stimulus)):
        #membrane voltage (integrate & fire)
        v_mem += (v_base - v_mem + v_offset + stimulus[i]) / mem_tau * deltat 
        #print(v_mem)
        
        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)

    return np.array(spike_times)


def whitenoise(cflow, cfup, dt, duration, rng=np.random):
     """Band-limited white noise.

     Generates white noise with a flat power spectrum between `cflow` and
     `cfup` Hertz, zero mean and unit standard deviation.  Note, that in
     particular for short segments of the generated noise the mean and
     standard deviation can deviate from zero and one.

     Parameters
     ----------
     cflow: float
         Lower cutoff frequency in Hertz.
     cfup: float
         Upper cutoff frequency in Hertz.
     dt: float
         Time step of the resulting array in seconds.
     duration: float
         Total duration of the resulting array in seconds.

     Returns
     -------
     noise: 1-D array
         White noise.
     """
     # next power of two:
     n = int(duration//dt)
     nn = int(2**(np.ceil(np.log2(n))))
     # draw random numbers in Fourier domain:
     inx0 = int(np.round(dt*nn*cflow))
     inx1 = int(np.round(dt*nn*cfup))
     if inx0 < 0:
         inx0 = 0
     if inx1 >= nn/2:
         inx1 = nn/2
     sigma = 0.5 / np.sqrt(float(inx1 - inx0))
     whitef = np.zeros((nn//2+1), dtype=complex)
     if inx0 == 0:
         whitef[0] = rng.randn()
         inx0 = 1
     if inx1 >= nn//2:
         whitef[nn//2] = rng.randn()
         inx1 = nn//2-1
     m = inx1 - inx0 + 1
     whitef[inx0:inx1+1] = rng.randn(m) + 1j*rng.randn(m)
     # inverse FFT:
     noise = np.real(np.fft.irfft(whitef))[:n]*sigma*nn
     return noise


def calculate_isi_frequency(spiketimes, t):
    """
    Calculate ISI frequency
    Do the following: either calculate the instantaneous fire rate over multiple trials, or use ISI 
    For ISI: the spike frequency is the same between each spike timepoints. For smoothing averaging will be done
    For now ISI is used, maybe other one can also be implemented soon.
    
    Parameters
    -----------
    spikes: 1D array
        Spike time points in seconds
    t: 1D array
        The time stamps in seconds
        Returns
    ----------
    freq: 1D array
        Frequency trace which starts at the time of first spike and ends at the time of the last spike.
    """
    spikeISI = np.diff(spiketimes)
    freq = 1/spikeISI
    freqtime = np.zeros(len(t))
    freqtime[0:np.squeeze(np.where(t==spiketimes[0]))]=freq[0]#initialize the frequency as 1/isi for the first spike (from onset on)
    for i in range(len(freq)):
        tbegin = int(np.where(t==spiketimes[i])[0])
        try:
            tend = int(np.where(t==spiketimes[i]+spikeISI[i])[0])
        except TypeError:
            freqtime[tbegin:] = freq[i]
            return freqtime
        freqtime[tbegin:tend] = freq[i]
    if spiketimes[-1] < t[-1]-0.5: #if last spike does not occur in the last 500 ms, set the firing rate to zero.
        freqtime[tend:] = 0
    else:
        freqtime[tend:] = freq[i]
    return freqtime


def spike_gauss_kernel(sigma, lenfactor, resolution):
    """
    The Gaussian kernel for spike convolution
    
    Parameters
    ----------
    sigma: float
        The kernel width in s
    lenfactor: float
        The size of the kernel in terms of sigma
    resolution: float
        The time resolution in s. Keep it same with the stimulus resolution
    
    Returns
    -------
    kernel: 1D array
        The Gaussian convolution kernel
    t: 1D array
        Time window of the kernel
    """
    t = np.arange(-sigma*lenfactor/2, sigma*lenfactor/2+resolution, resolution) 
    #t goes from -sigma/2*lenfactor to +sigma/2*lenfactor, +resolution because arange stops prematurely. 
    #the start and stop point of t is irrelevant, but only kernel is used
    kernel = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-((t)**2) / (2*sigma**2)) 
    #maximum of the Gaussian kernel is in the middle
    return kernel, t


def convolved_spikes(spiketimes, t, kernel):
    """
    Convolve the spikes with a given kernel
    
    Parameters
    ----------
    spiketimes: 1-D array
        The array containing spike occurence times (in seconds)
    t: 1-D array
        The time array in seconds
    kernel: 1-D array
        The kernel array
        
    Returns
    --------
    convolvedspikes: 1-D array
        The array containing convolved spikes
    spikearray: 1-D array
        The logical array containing 1 for the time point where spike occured.
    """
    #run the model for the given stimulus and get spike times
    #spike train with logical 1s and 0s
    spikearray = np.zeros(len(t)) 
    #convert spike times to spike trains
    spikearray[(spiketimes//(t[1]-t[0])).astype(np.int)] = 1
    
    #spikearray[np.digitize(spiketimes,t)-1] = 1 #np.digitize(a,b) returns the index values for a where a==b. For convolution
    #for np.digitize() see https://numpy.org/doc/stable/reference/generated/numpy.digitize.html is this ok to use here?
    #np.digitize returns the index as if starting from 1 for the last index, so -1 in the end THIS stays just in case FYI
    
    #convolve the spike train with the gaussian kernel    
    convolvedspikes = np.convolve(kernel, spikearray, mode='same')
    return convolvedspikes, spikearray


def decibel_transformer(power):
    """
    Transform power to decibel (0 dB is the maximum value in power data)
    
    Parameters
    ----------
    power: 1-D array
        The array of power values to be transformed into decibel
        
    Returns
    -------
    dB: 1-D array
        Decibel transformed power
    """ 
    dB = 10.0*np.log10(power/np.max(power))   # power to decibel
    return dB


class power_spectra_transfer_funcs:
    """
    Functions concerning input-output relationships (stimulus/response power spectra, transfer functions, coherence, lower bound info)
    
    Parameters
    -----------
    Stimulus: 1-D array
        The stimulus array
    spiketimes: 1-D array
        The array containing spike times
    t: 1-D array
        The time array
    kernel: 1-D array
        Array of the convolution kernel
    nperseg: float
        Power spectrum number of datapoints per segment
    SAM: boolean
        If True, calculation is done in functions for SAM stimulus, else for RAM stimulus
    """
    
    
    def __init__(self, stimulus, spiketimes, t, kernel, nperseg):
        self.stimulus = stimulus
        self.spiketimes = spiketimes
        self.t = t
        self.kernel = kernel
        self.nperseg = nperseg
        self.t_delta = self.t[1]-self.t[0]
        self.convolvedspikes, self.spikearray = convolved_spikes(self.spiketimes, self.t, self.kernel)

        
    def power_spectrum(self):
        """
        Calculate power spectrum for given cell and stimulus. Note that first 100 ms of spike trains is discarded.
        
        Returns
        --------
        pr: 1-D array
            The array of response frequency powers
        ps: 1-D array
            The array of stimulus frequency powers
        fr: 1-D array
            The array of response power spectrum frequencies
        fs: 1-D array
            The array of stimulus power spectrum frequencies
        """   
        #stimulus
        fs, ps = welch(self.stimulus[self.t>0.1], nperseg=self.nperseg, fs=1/self.t_delta)    

        #response
        fr, pr = welch(self.convolvedspikes[self.t>0.1], nperseg=self.nperseg, fs=1/self.t_delta)
        return pr, ps, fr, fs


    def cross_spectral_density(self, calcoherence=False):
        """
        Calculate cross spectral density and (possibly) stimulus-response coherence for given cell and stimulus. 
        Note that first 100 ms of spike train is discarded.
        
        Parameters
        ----------
        calcoherence: logical
            If True, the coherence is also calculated for the given stimulus and model parameters.
        
        Returns
        --------
        f: 1-D array
            The array of power spectrum frequencies
        psr: 1-D array
            The array of cross spectral density power
        fcoh: 1-D array
            The array of frequencies for coherence
        gamma: 1-D array
            Coherence between stimulus and response (0-1, 1 means noiseless perfect linear system.)
        """
        #run the model for the given stimulus and get spike times
        #spiketimes, spikeISI, meanspkfr = stimulus_ISI_calculator(cellparams, stimulus, tlength=len(t)*t_delta)
            
        f, psr = csd(self.convolvedspikes[self.t>0.1], self.stimulus[self.t>0.1], nperseg=self.nperseg, fs=1/self.t_delta)
        if calcoherence == True:
            fcoh, gamma = coherence(self.convolvedspikes[self.t>0.1], self.stimulus[self.t>0.1], \
                                    nperseg=self.nperseg, fs=1/self.t_delta)
            return f, psr, fcoh, gamma
        else:
            return f, psr
        
    def interpolate_power_spectrum_value(self, frequency, fs, ps, fr, pr):
        """
        Interpolate the stimulus and response powers for a given frequency value. Useful for transfer function 
        calculation of SAM stimulus. Interpolation done by scipy.interpolate.interp1d
        
        Parameters
        ----------
        frequency: float
            The frequency value at which stimulus and response power spectral densities should be interpolated.
        fs: 1-D array
            The array of stimulus power spectrum frequencies
        ps: 1-D array
            The array of stimulus frequency powers
        fr: 1-D array
            The array of response power spectrum frequencies
        pr: 1-D array
            The array of response frequency powers
            
        Returns
        --------
        psi: float
            Interpolated stimulus power spectral density value
        pri: float
            Interpolated response power spectral density value
        """
        pr_interpolator = interpolate(fr, pr)
        pri = pr_interpolator(frequency) #interpolated response power at given frequency
        ps_interpolator = interpolate(fs, ps)
        psi = ps_interpolator(frequency) #interpolated stimulus power at given frequency
        return psi, pri