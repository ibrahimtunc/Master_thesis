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
from matplotlib import colors as cl
import shelve


"""
Master thesis package script.
ADD HERE THE LIST OF ALL FUNCTIONS AND CLASSES WITH SHORT DESCRIPTIONS
"""

@jit(nopython=True)
def LIF_reduced(stimulus, v_zero=0.0, v_base=0.0, v_offset=1.5, mem_tau=0.015, threshold=1.0, deltat=0.00005,
                noise_strength=0.0):
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
    noise_strength: float
        Strength of the noise with zero mean. This value determines the noise standard deviation. Default is 
        noiseless. Burkitt 2006 => this value is equal to sigma*sqrt(2*tau) where sigma is intensity coefficient.
    Returns
    -------
    spike_times: 1-D array
        Simulated spike times in seconds.
    """  
    #LIFAC reduced to passive membrane linear LIF model -> No dendritic compartment, no refractory perid, no 
    #adaptation for now. Also dscard noise in the very first step.

    #initial conditions:
    v_mem = v_zero

    #rectify stimulus array:
    stimulus = stimulus.copy()
    stimulus[stimulus < 0.0] = 0.0

    #prepare noise
    noise = np.random.randn(len(stimulus))
    noise *= noise_strength / np.sqrt(deltat) #scale white noise with square root of time step, since else noise
                                              #is time dependent, this makes it time step invariant.    

    #integrate:
    spike_times = []
    for i in range(len(stimulus)):
        #membrane voltage (integrate & fire)
        v_mem += (v_base - v_mem + v_offset + stimulus[i] + noise[i]) / mem_tau * deltat 
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


class power_spectra_cross_spectrum:
    """
    Calculate power spectrum of stimulus and response as well as the cross spectral density. Additional function for 
    SAM to interpolate values for the frequency of interest
    """
    
    
    def __init__(self, stimulus, spiketimes, t, kernel, nperseg):
        """
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
        """
        t_delta = t[1]-t[0]
        self.convolvedspikes, self.spikearray = convolved_spikes(spiketimes, t, kernel)
        #stimulus
        self.fs, self.ps = welch(stimulus[t>0.1], nperseg=nperseg, fs=1/t_delta)    

        #response
        self.fr, self.pr = welch(self.convolvedspikes[t>0.1], nperseg=nperseg, fs=1/t_delta)
                    
        self.frs, self.prs = csd(self.convolvedspikes[t>0.1], stimulus[t>0.1], 
                                 nperseg=nperseg, fs=1/t_delta)
        
        
    def interpolate_power_spectrum_value(self, frequency):
        """
        Interpolate the stimulus and response powers for a given frequency value. Useful for transfer function 
        calculation of SAM stimulus. Interpolation done by scipy.interpolate.interp1d
        
        Parameters
        ----------
        frequency: float
            The frequency value at which stimulus and response power spectral densities should be interpolated.
            
        Returns
        --------
        psi: float
            Interpolated stimulus power spectral density value
        pri: float
            Interpolated response power spectral density value
        """
        pr_interpolator = interpolate(self.fr, self.pr)
        pri = pr_interpolator(frequency) #interpolated response power at given frequency
        ps_interpolator = interpolate(self.fs, self.ps)
        psi = ps_interpolator(frequency) #interpolated stimulus power at given frequency
        prs_interpolator = interpolate(self.frs, self.prs)
        prsi = prs_interpolator(frequency)
        return psi, pri, prsi
        
    
class simulation:
    """
    Run a simulation for the given stimulus settings. 
    This class returns the required parameters for spectral densities
    """
    
    
    def __init__(self, nsimul, stimulus, t, kernel, nperseg, **kwargs):
        """
        Parameters
        ----------
        nsimul : int
            Number of simulations. Note that in each trial stimulus is the same, this parameter is therefore used for 
            the noisy model.
        stimulus : 1-D array
            Stimulus used in the simulation.
        t : 1-D array
            Time array.
        kernel : 1-D array
            Convolution kernel array.
        nperseg : int
            Number of datapoints per segment in power spectrum (welch).
        **kwargs : 
            Additional keyword arguments to change the model settings (see function LIF_reduced).
    
        """
        if nsimul == 1:
            spkt = LIF_reduced(stimulus, **kwargs)
            self.ios = power_spectra_cross_spectrum(stimulus, spkt, t, kernel, nperseg) 
            self.frs = len(spkt) / (t[-1]-t[0])
        
        else:
            self.ios = []
            self.frs = np.zeros(nsimul)
            for idx in range(nsimul):
                spkt = LIF_reduced(stimulus, **kwargs)
                io = power_spectra_cross_spectrum(stimulus, spkt, t, kernel, nperseg)
                self.ios.append(io)
                self.frs[idx] = len(spkt) / (t[-1]-t[0])
            

class coherence_and_transfer_func:
    """
    Functions to calculate coherence and transfer function
    
    Functions
    ---------
    -calculate_transfer_func
    -calculate_sr_coherence
    """
    
    def __init__(self, prss, psss, prrs):
        self.prss = prss
        self.psss = psss
        self.prrs = prrs
        
        
    def calculate_transfer_func(self):
        """
        Calculate the transfer function for the given csd and stimulus power spectrum arrays
    
        Parameters
        ----------
        prss : 2-D (or 1-D) array
            Cross spectral density array. If array is 2-D, first dimension is simulation ID
        psss : 2-D (or 1-D) array
            Stimulus power spectrum array. If array is 2-D, first dimension is simulation ID
    
        Returns
        -------
        tf : 1-D array
            Transfer function array. If prss and psss are 2-D arrays, returns the average transfer function.
    
        """
        if len(self.prss.shape) == 1:
            return np.abs(self.prss / self.psss)
        else:
            return np.abs(np.mean(self.prss,axis=0) / np.mean(self.psss, axis=0))
        
    
    def calculate_sr_coherence(self):
        """
        Calculate stimulus response coherence
    
        Returns
        -------
        gammasq : 1-D array
            Stimulus-response coherence array
        """
        if len(self.prss.shape) == 1:
            return np.abs(self.prss)**2 / (self.prrs * self.psss)
        else:
            return np.abs(np.mean(self.prss, axis=0))**2 / (np.mean(self.prrs, axis=0) * np.mean(self.psss, axis=0))
        

def coefficient_of_variation(spiketimes):
    """
    Calculate the coefficient of variation for the given spike times

    Parameters
    ----------
    spiketimes : 1-D array
        Array containing spike times.

    Returns
    -------
    cv : float
        Coefficient of variation defined as std(ISI)/mean(ISI) where ISI is the interspike interval.
    """
    ISI = np.diff(spiketimes)
    return np.std(ISI) / np.mean(ISI)
  
    
def model_fr_spk_per_cyc(ncycles, fAMs, contrasts, dt, noise_strength, **kwargs):
    """
    Compute the average firing rate and number of spikes per cycle for the given sine wave settings.

    Parameters
    ----------
    ncycles : int
        Number of cycles in the stimulus. Stimulus length is wave period times ncycles.
    fAMs : 1-D array
        The sine wave frequency array.
    contrasts : 1-D array
        Array of sine wave amplitudes.
    dt : float
        Integration time length of the LIF model.
    noise_strength : float
        Noise strength of the LIF model.
    **kwargs : 
        Additional keyword arguments for the LIF model. See function LIF_reduced.

    Returns
    -------
    fr : 2-D array
        Average firing rate array. Shape is len(fAMs) x len(contrasts)
    spkpercyc : Average number of spikes per cycle (calculated in a cycle by cycle manner).
                Shape is len(fAMs) x len(contrasts).
    """
    #preallocate firing rate array
    fr = np.zeros([len(fAMs), len(contrasts)]) #average firing rate
    spkpercyc = np.zeros([len(fAMs), len(contrasts)]) #average number of spikes per cycle
    for fidx, fAM in enumerate(fAMs):
        #prepare the sine wave -> single cycle
        tcyc = np.arange(0,1/fAM, dt) #time required for a single cycle.
        sincyc = np.sin(2*np.pi*fAM*tcyc) #single sine wave cycle
        #generate the whole stimulus 
        sinwave = np.tile(sincyc,ncycles)
        
        for cidx, contrast in enumerate(contrasts):
            stim = sinwave * contrast #generate the stimulus
            spkt = LIF_reduced(stim, deltat=dt, noise_strength=noise_strength, **kwargs) #spike times
            fr[fidx, cidx] = len(spkt) / (len(sinwave)*dt) #average firing rate over all cycles
            
            #look at the number of spikes per cycle and take the average
            tprev = 0 #previous time stamp 
            nspkcyc = np.zeros(ncycles) #number of spikes per cycle
            for nidx in range(ncycles):
                tnow = (nidx+1) / fAM #current time stamp
                nspkcyc[nidx] = len(spkt[(spkt>=tprev) & (spkt<tnow)])
                #print(tprev, tnow)
                tprev = tnow
            spkpercyc[fidx, cidx] = np.mean(nspkcyc)
        print( '%% %.3f complete.'%(100 * ( (fidx+1) / len(fAMs) )) ) 
    return fr, spkpercyc


#Add further functions here    
#...


#PLOTTING
#General settings
figdict = {'axes.titlesize' : 25,
           'axes.labelsize' : 25,
           'xtick.labelsize' : 25,
           'ytick.labelsize' : 25,
           'legend.fontsize' : 25,
           'figure.titlesize' : 30,
           'image.cmap' : 'gray'}
plt.style.use(figdict)
  
#Functions bundled together in a class    
class plotter:
    """
    Class for general plotting functions.
    
    Subclasses
    -----------
    1) stimulus_response_plots: Plots dealing with stimulus response relationships (transfer funcs etc.)
        
        Functions
        ----------    
        -pr_plotter
        -csd_plotter
        -transfer_func_plotter
        -s_r_coherence_plotter
        -pr_freq_plotter
    
    1) fr_spkpercyc: Plots dealing with average firing rate and number of spikes per cycle for a given sine wave
        
        Functions
        ----------
        -color_plot_rf_spkpercyc
        -fr_contrast_subplots_for_f
        -fr_f_subplots_for_contrasts
        -avgspk_contrast_subplots_for_f
        -avgspk_f_subplots_for_contrasts
    """
    
    
    class stimulus_response_plots:
        """
        Plots dealing with stimulus-response relationships such as power spectra, transfer functions, coherence, csd.
        """
        
        def __init__(self, nrows, ncols, sadj, frange, bfr, sfr, contrasts, SAM):
            """
            General plot parameters
    
            Parameters
            ----------
            nrows : int
                Number of rows in the subplot.
            ncols : int
                Number of columns in the subplot.
            sadj : dict
                Dictionary for subplots_adjust.
            frange : 1-D array
                Frequency range.
            bfr : float
                Baseline firing rate.
            sfr : 1-D array
                Firing rate elicited by the stimulus. Shape len(contrasts)
            contrasts : 1-D array
                Contrasts used for the stimulus.
            SAM : boolean
                If True, plot color is blue (for SAM). If False, plot color is red (for RAM)
                
            Returns
            -------
            None.
    
            """
            self.nrows = nrows
            self.ncols = ncols
            self.sadj = sadj
            self.frange = frange
            self.bfr = bfr
            self.contrasts = contrasts
            self.xticks = [0, np.round(np.ceil(np.max(self.frange)),-1)]
            if SAM == True:
                self.colstr = 'b'
            else:
                self.colstr = 'r'
        
        
        def pr_plotter(self, pr):
            """
            Plot response power for different contrasts in subplots
    
            Parameters
            ----------
            pr : 2-D array
                Response power array. Shape len(contrasts) x len(frange)
    
            Returns
            -------
            fig : Figure object
                Current figure object
            """
            
            fig, axs = plt.subplots(self.nrows, self.ncols, sharex=True, sharey='row')
            axs[np.int(self.nrow//2), 0].set_ylabel('Power ' r'[$\frac{Hz^2}{Hz}$]') #!these labels are adapted for 5x5 subplot.
            axs[-1, np.int(self.ncol//2)].set_xlabel('Frequency [Hz]')
            axs = axs.flatten()
            skipidx = len(self.contrasts) // len(axs)
            for pidx, ax in enumerate(axs):
                pidx *= skipidx
                ax.set_title('[%.4f]' %(self.contrasts[pidx]))
                ax.set_xticks(np.linspace(*self.xticks, 5))
                ax.plot(self.frange, pr[pidx], self.colstr+'-', label='response')
                ax.plot([self.bfr, self.bfr], [0,np.max(pr[pidx])], 
                        'k--', label='baseline')
                ax.plot([self.sfr[pidx], self.sfr[pidx]], [0,np.max(pr[pidx])], 
                        'k.-', label='contrast avg')
            plt.subplots_adjust(**self.sadj)
            return fig
        
        
        def csd_plotter(self, prs):
            """
            Plot the cross spectral density for different contrasts in subplots
    
            Parameters
            ----------
            prs : 2-D array
                Cross spectral density array. Shape len(contrasts) x len(frange).
    
            Returns
            -------
            fig : Figure object
                Current figure object.
            """
            
            fig, axs = plt.subplots(self.nrows, self.ncols, sharex=True, sharey='row')
            axs[np.int(self.nrow//2), 0].set_ylabel('Power ' r'[$\frac{Hz^2}{Hz}$]') #adjusted for any subplot size
            axs[-1, np.int(self.ncol//2)].set_xlabel('Frequency [Hz]')
            axs = axs.flatten()
            skipidx = len(self.contrasts) // len(axs)

            for pidx, ax in enumerate(axs):
                pidx *= skipidx
                ax.set_title('[%.4f]' %(self.contrasts[pidx]))
                ax.set_xticks(np.linspace(*self.xticks, 5))
                ax.plot(self.frange, np.abs(prs[pidx]), self.colstr+'-', label='csd')
                ax.plot([self.bfr, self.bfr], [0,np.max(np.abs(prs[pidx]))], 
                        'k--', label='baseline')
                ax.plot([self.sfr[pidx], self.sfr[pidx]], [0,np.max(np.abs(prs[pidx]))], 
                        'k.-', label='contrast avg')
            #ax.legend()
            plt.subplots_adjust(**self.sadj)
            return fig
        
        
        def transfer_func_plotter(self, tfs):
            """
            Plot the transfer function for different contrasts into subplots.
    
            Parameters
            ----------
            tfs : 2-D array
                Transfer function array. Shape len(contrasts) x len(frange).
            
            Returns
            -------
            fig: Figure object.
                Current figure object
            """
            
            fig, axs = plt.subplots(self.nrows, self.ncols, sharex=True, sharey='row')
            axs[np.int(self.nrow//2), 0].set_ylabel('Gain ' r'[$\frac{Hz}{mV}$]')
            axs[-1, np.int(self.ncol//2)].set_xlabel('Frequency [Hz]')
            axs = axs.flatten()
            skipidx = len(self.contrasts) // len(axs)

            for pidx, ax in enumerate(axs):
                pidx *= skipidx
                ax.set_title('[%.4f]' %(self.contrasts[pidx]))
                ax.set_xticks(np.linspace(*self.xticks, 5))
                ax.plot(self.frange, tfs[pidx], self.colstr+'-', label='gain')
                ax.plot([self.bfr, self.bfr], [0,np.max(tfs[pidx])], 
                        'k--', label='baseline')
                ax.plot([self.sfr[pidx], self.sfr[pidx]], [0,np.max(tfs[pidx])], 
                        'k.-', label='contrast avg')
            #ax.legend()
            plt.subplots_adjust(**self.sadj)
            return fig
        
        
        def s_r_coherence_plotter(self, coh):
            """
            Plot the stimulus-response coherence for the given contrasts into subplots.
    
            Parameters
            ----------
            coh : 2-D array
                Stimulus-response coherence array. Shape len(contrasts) x len(frange).
            
            Returns
            -------
            fig: Figure object.
                Current figure object
            """
            
            fig, axs = plt.subplots(self.nrows, self.ncols, sharex=True, sharey='row')
            axs[np.int(self.nrow//2), 0].set_ylabel(r'$\gamma^2_{SR}$') #adjusted for any size
            axs[-1, np.int(self.ncol//2)].set_xlabel('Frequency [Hz]')
            axs = axs.flatten()
            skipidx = len(self.contrasts) // len(axs)

            for pidx, ax in enumerate(axs):
                pidx *= skipidx
                ax.set_title('[%.4f]' %(self.contrasts[pidx]))
                ax.set_xticks(np.linspace(*self.xticks, 5))
                ax.plot(self.frange, coh[pidx], self.colstr+'-', label=r'$\gamma^2$')
                ax.plot([self.bfr, self.bfr], [0,np.max(coh[pidx])], 
                        'k--', label='baseline')
                ax.plot([self.sfr[pidx], self.sfr[pidx]], [0,np.max(coh[pidx])], 
                        'k.-', label='contrast avg')
            #ax.legend()
            plt.subplots_adjust(**self.sadj)
            return fig
        
        
        def pr_freq_plotter(self, pr):
            """
            Plot response powers as a function of contrast for different frequencies


            Parameters
            ----------
            pr : 2-D array
                Response power array. Shape len(contrasts) x len(frange)
    
            Returns
            -------
            fig : Figure object
                Current figure object

            """
            fig, axs = plt.subplots(self.nrows, self.ncols, sharex=True, sharey='row')
            axs[np.int(self.nrow//2), 0].set_ylabel('Power ' r'[$\frac{Hz^2}{Hz}$]') #adjusted for any size
            axs[-1, np.int(self.ncol//2)].set_xlabel('Contrast [%]')
            axs = axs.flatten()
    
            skipidx = len(self.frange[1:]) // len(axs)
            for pidx, ax in enumerate(axs):
                plotnum = pidx*skipidx
                ax.set_title('[%.4f]' %(self.frange[plotnum]))
                ax.semilogx(self.contrasts, pr[:,plotnum], self.colstr+'-')
                ax.set_xticks(np.logspace(np.log10(np.min(self.contrasts)),np.log10(np.max(self.contrasts)),4))
                ax.set_xticklabels(np.round(np.logspace(np.log10(np.min(self.contrasts)),
                                                        np.log10(np.max(self.contrasts)),4),2))
            plt.get_current_fig_manager().window.showMaximized()
            return fig
    
    
    class fr_spkpercyc:
        """
        Plots dealing with average firing rate and average spike number per cycle
        """
        
        def __init__(self, fr, fAMs, contrasts, spkpercyc):
            """
            General class parameters
            
            Parameters
            ----------
            ncycles : int
                Number of cycles in the stimulus. Stimulus length is wave period times ncycles.
            fAMs : 1-D array
                The sine wave frequency array.
            contrasts : 1-D array
                Array of sine wave amplitudes.
            dt : float
                Integration time length of the LIF model.
            noise_strength : float
                Noise strength of the LIF model.
            **kwargs : 
                Additional keyword arguments for the LIF model. See function LIF_reduced.
            """
            self.fr = fr
            self.fAMs = fAMs
            self.contrasts = contrasts
            self.spkpercyc = spkpercyc
    
    
        def color_plot_rf_spkpercyc(self):
            """
            Color plots for the average firing rate and spike per cycle using the results from function
            model_fr_spk_per_cyc
        
            Parameters
            ----------
            fr : 2-D array
                Average firing rate array. Shape is len(fAMs) x len(contrasts)
            fAMs : 1-D array
                The sine wave frequency array.
             contrasts : 1-D array
                Array of sine wave amplitudes.
             spkpercyc : Average number of spikes per cycle (calculated in a cycle by cycle manner).
                         Shape is len(fAMs) x len(contrasts).
        
            Returns
            -------
            axs : 1-D array
                Axes array.
            """
            
            fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
            im0 = axs[0].imshow(self.fr.T, cmap='jet', origin='lower', 
                                extent=[0,len(self.fAMs),0,len(self.contrasts)])
            axs[0].set_xticks(np.round(np.linspace(np.min(self.fAMs), np.max(self.fAMs), 5)))
            axs[0].set_yticks(np.linspace(0, len(self.contrasts), 5))
            axs[0].set_yticklabels(np.round(np.linspace(np.min(self.contrasts), np.max(self.contrasts), 5),3))
            plt.colorbar(im0, ax=axs[0])
            im1 = axs[1].imshow(self.spkpercyc.T, cmap='jet', origin='lower', 
                                extent=[0,len(self.fAMs),0,len(self.contrasts)], norm=cl.LogNorm())
            axs[1].set_xticks(np.round(np.linspace(np.min(self.fAMs), np.max(self.fAMs), 5)))
            axs[1].set_yticks(np.linspace(0, len(self.contrasts), 5))
            axs[1].set_yticklabels(np.round(np.linspace(np.min(self.contrasts), np.max(self.contrasts), 5),3))
            cb = plt.colorbar(im1, ax=axs[1], ticks=np.logspace(np.log10(0.1), np.log10(np.max(self.spkpercyc)), 10))
            cb.ax.set_yticklabels(np.round(np.logspace(np.log10(0.1), np.log10(np.max(self.spkpercyc)), 10),2))
            axs[0].set_ylabel('Contrast [%]')
            axs[0].set_xlabel('Stimulus frequency [Hz]')
            return axs


        def fr_contrast_subplots_for_f(self, nrow, ncol):
            """
            Plot average firing rates as a function of contrast for different sine wave frequencies

            Parameters
            ----------
            nrow : int
                Number of rows in the subplot.
            ncol : int
                Number of cols in the subplot.

            Returns
            -------
            fig : Figure object
                Current figure object.
            """
            fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
            axs[np.int(nrow//2), 0].set_ylabel('Firing rate [Hz]')
            axs[-1, np.int(ncol//2)].set_xlabel('Contrast [%]')
            axs = axs.flatten()
            skipidx = self.fr.shape[0] // len(axs)
            for idx, ax in enumerate(axs):
                ax.plot(self.contrasts, self.fr[skipidx*idx, :], 'k-')
                ax.set_title(np.round(self.fAMs[skipidx*idx],3))
            return fig
        
        
        def fr_f_subplots_for_contrasts(self, nrow, ncol):
            """
            Plot average firing rates as a function of sine wave frequencies for different contrasts

            Parameters
            ----------
            nrow : int
                Number of rows in the subplot.
            ncol : int
                Number of cols in the subplot.

            Returns
            -------
            fig : Figure object
                Current figure object.
            """
            fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
            axs[np.int(nrow//2), 0].set_ylabel('Firing rate [Hz]')
            axs[-1, np.int(ncol//2)].set_xlabel('Stimulus frequency [Hz]')
            axs = axs.flatten()
            skipidx = self.fr.shape[1] // len(axs)
            for idx, ax in enumerate(axs):
                ax.plot(self.fAMs, self.fr[:, skipidx*idx], 'k-')
                ax.set_title(np.round(self.contrasts[skipidx*idx],3))
            return fig
        
        
        def avgspk_contrast_subplots_for_f(self, nrow, ncol):
            """
            Plot average number of spikes per cycle as a function of contrast for different sine wave frequencies

            Parameters
            ----------
            nrow : int
                Number of rows in the subplot.
            ncol : int
                Number of cols in the subplot.

            Returns
            -------
            fig : Figure object
                Current figure object.
            """
            fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
            axs[np.int(nrow//2), 0].set_ylabel('Spike number per cycle')
            axs[-1, np.int(ncol//2)].set_xlabel('Contrast [%]')
            axs = axs.flatten()
            skipidx = self.spkpercyc.shape[0] // len(axs)
            for idx, ax in enumerate(axs):
                ax.plot(self.contrasts, self.spkpercyc[skipidx*idx, :], 'k-')
                ax.set_title(np.round(self.fAMs[skipidx*idx],3))
            return fig


        def avgspk_f_subplots_for_contrasts(self, nrow,ncol):
            """
            Plot average number of spikes per cycle as a function of sine wave frequencies for different contrasts

            Parameters
            ----------
            nrow : int
                Number of rows in the subplot.
            ncol : int
                Number of cols in the subplot.

            Returns
            -------
            fig : Figure object
                Current figure object.
            """
            fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey='row')
            axs[np.int(nrow//2), 0].set_ylabel('Spike number per cycle')
            axs[-1, np.int(ncol//2)].set_xlabel('Stimulus frequency [Hz]')
            axs = axs.flatten()
            skipidx = self.spkpercyc.shape[1] // len(axs)
            for idx, ax in enumerate(axs):
                ax.plot(self.fAMs, self.spkpercyc[:, skipidx*idx], 'k-')
                ax.set_title(np.round(self.contrasts[skipidx*idx],3))
            return fig
        

class file_management:
    """
    Functions used to save and load simulation data and other files. 
    Source: https://stackoverflow.com/questions/2960864/how-to-save-all-the-variables-in-the-current-python-session
    
    Functions
    ---------
    -save_session
    -load_session
    """
    
    def __init__(self, current_dir):
        """
        Parameters
        ----------
        current_dir : str
            Current directory to save/load files.
        """
        self.current_dir = current_dir
        
    
    def save_file(self, filename):
        """
        Save the current session.

        Parameters
        ----------
        filename : str
            Name of the file. Describe here the specifics of the current session (parameters used etc.)

        Returns
        -------
        None.
        """
        savename = self.current_dir + '\\' + filename + '.out'
        my_shelf = shelve.open(savename,'n') # 'n' for new

        for key in dir():
            try:
                my_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        print('Save complete at %s!' %(savename))
        my_shelf.close()
        return
    
    
    def load_file(self, filename):
        """
        Load a previous session.

        Parameters
        ----------
        filename : str
            Name of the file. For copy-paste ease raw file name (no extension, no directory) is used.

        Returns
        -------
        None.
        """
        loadname = self.current_dir + '\\' + filename + '.out'
        my_shelf = shelve.open(loadname)
        for key in my_shelf:
            globals()[key]=my_shelf[key]
        print('Previous session %s is loaded!' %(filename))
        my_shelf.close()
