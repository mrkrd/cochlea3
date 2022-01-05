# cython: language_level=3

from math import ceil, floor

import numpy as np
import scipy.signal as dsp

from cochlea3.zilany2014.helper import ffGn

from libc.stdlib cimport malloc
cimport numpy as np


cdef extern from "stdlib.h":
    void *memcpy(void *str1, void *str2, size_t n)


cdef extern from "model_IHC.h":
    void IHCAN(
        double *px,
        double cf,
        int nrep,
        double tdres,
        int totalstim,
        double cohc,
        double cihc,
        int species,
        double *ihcout
    )


cdef extern from "model_Synapse.h":
    double Synapse(
        double *ihcout,
        double tdres,
        double cf,
        int totalstim,
        int nrep,
        double spont,
        double noiseType,
        double implnt,
        double sampFreq,
        double *synouttmp,
        double *randNums
    )
    int SpikeGenerator(
        double *synouttmp,
        double tdres,
        int totalstim,
        int nrep,
        double *sptime
    )


cdef extern from "Python.h":
    ctypedef int Py_intptr_t


cdef extern from "numpy/arrayobject.h":
    ctypedef Py_intptr_t npy_intp
    object PyArray_SimpleNewFromData(
        int nd,
        npy_intp* dims,
        int typenum,
        void* data
    )


np.import_array()


def run_ihc(
        np.ndarray[np.float64_t, ndim=1] sound,
        double cf,
        double fs,
        species='cat',
        double c_ohc=1.,
        double c_ihc=1.
):
    """Run middle ear filter, BM filters and IHC model.

    Parameters
    ----------
    sound : ndarray
        Output of the middle ear filter in Pascal.
    cf : float
        Characteristic frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    species : {'cat', 'human', 'human_glasberg1990'}
        Species.
    c_ihc, c_ohc : float
        Degeneration parameters for IHC and OHC cells.

    Returns
    -------
    ndarray
        IHC receptor potential.

    """
    if species == 'cat' and ((cf < 125) or (cf >= 40e3)):
        raise RuntimeError(f"Invalid value of CF for {species}: {cf}")
    elif species == 'human' and ((cf < 125) or (cf > 20e3)):
        raise RuntimeError(f"Invalid value of CF for {species}: {cf}")

    if fs < 100e3 or fs > 500e3:
        raise RuntimeError(f"Invalid value of FS: {fs}")
    if c_ohc < 0 or c_ohc > 1:
        raise RuntimeError(f"Invalid value of c_ohc: {c_ohc}")
    if c_ihc < 0 or c_ihc > 1:
        raise RuntimeError(f"Invalid value of c_ihc: {c_ihc}")

    species_id = {
        'cat': 1,
        'human': 2,
        'human_glasberg1990': 3,
    }[species]

    # Input sound
    if not sound.flags['C_CONTIGUOUS']:
        sound = np.numpy.ascontiguousarray(sound)
    cdef double *sound_data = <double *>np.PyArray_DATA(sound)

    # Output IHC voltage
    v_ihc = np.zeros(len(sound))
    cdef double *v_ihc_data = <double *>np.PyArray_DATA(v_ihc)

    IHCAN(
        sound_data,             # px
        cf,                     # cf
        1,                      # nrep
        1.0/fs,                 # tdres
        len(sound),             # totalstim
        c_ohc,                  # cohc
        c_ihc,                  # cihc
        species_id,             # species
        v_ihc_data,             # ihcout
    )

    return v_ihc


def run_synapse(
        np.ndarray[np.float64_t, ndim=1] v_ihc,
        double fs,
        double cf,
        anf_type='hsr',
        powerlaw='actual',
        seed=None,
):
    """Run a model of an IHC-AN synapse.

    Parameters
    ----------
    v_ihc : ndarray
        IHC receptor potential.
    fs : float
        Sampling frequency of ``v_ihc``.
    cf : float
        Characteristic frequency.
    anf_type : {'hsr', 'msr', 'lsr'}
        Type of an auditory nerve fiber.
    powerlaw : {'actual', 'approximate'}
        Implementation of the powerlaw.
    seed : int, optional
        Random seed for the factorial Gauss noise generator.
        ``None`` disables the noise.

    Returns
    -------
        Spiking probability.

    """
    assert (cf > 79.9) and (cf < 40e3), "Wrong CF: 80 <= cf < 40e3, CF = %s" % str(cf)
    assert (fs >= 100e3) and (fs <= 500e3), "Wrong Fs: 100e3 <= fs <= 500e3"
    assert anf_type in ['hsr', 'msr', 'lsr'], "anf_type not hsr/msr/lsr"
    assert powerlaw in ['actual', 'approximate'], "powerlaw not actual/approximate"

    spont_rate = {
        'lsr': 0.1,
        'msr': 4.0,
        'hsr': 100.0,
    }[anf_type]

    powerlaw_id = {
        'actual': 1,
        'approximate': 0
    }[powerlaw]

    # Input IHC voltage
    if not v_ihc.flags['C_CONTIGUOUS']:
        v_ihc = v_ihc.copy(order='C')
    cdef double *v_ihc_data = <double *>np.PyArray_DATA(v_ihc)

    # Output synapse data (spiking probabilities)
    p_spike = np.zeros_like(v_ihc)
    cdef double *p_spike_data = <double *>np.PyArray_DATA(p_spike)

    # ffGn
    fs_noise = 10e3
    delaypoint = floor(7500/(cf/1e3))
    N = ceil(
        (len(v_ihc)*1 + 2*delaypoint)*(1/fs)*samp_freq
    )
    noise = ffGn(
        N,
        1/fs_noise,
        0.9,
        noise_type,
        spont_rate,
    )
    if not noise.flags['C_CONTIGUOUS']:
        noise = np.ascontiguousarray(noise)
    cdef double *noise_data = <double *>np.PyArray_DATA(noise)

    # Run synapse model
    Synapse(
        v_ihc_data,             # ihcout
        1.0/fs,                 # tdres
        cf,                     # cf
        len(v_ihc),             # totalstim
        1,                      # nrep
        spont_rate,             # spont
        noise_type,             # noiseType (replaced by ffGn_data)
        powerlaw_id,            # implnt
        fs_noise,               # sampFreq
        p_spike_data,           # synouttmp
        noise_data              # randNums
    )

    return p_spike


def run_spike_generator(
        np.ndarray[np.float64_t, ndim=1] synout,
        double fs
):
    """Run spike generator.

    synout: synapse output
    fs: sampling frequency

    return: sptime

    """
    # Input IHC voltage
    if not synout.flags['C_CONTIGUOUS']:
        synout = synout.copy(order='C')
    cdef double *synout_data = <double *>np.PyArray_DATA(synout)

    # Output spikes (signal)
    sptimes = np.zeros(int(np.ceil(len(synout)/0.00075/fs)))
    cdef double *sptimes_data = <double *>np.PyArray_DATA(sptimes)

    # Run synapse model
    SpikeGenerator(
        synout_data,            # synouttmp
        1./fs,                  # tdres
        len(synout),            # totalstim
        1,                      # nprep
        sptimes_data            # sptime
    )

    spikes = np.array(sptimes[sptimes != 0])

    return spikes


# cdef public double* generate_random_numbers(long length):
#     arr = np.random.rand(length)

#     if not arr.flags['C_CONTIGUOUS']:
#         arr = arr.copy(order='C')

#     cdef double *data_ptr = <double *>np.PyArray_DATA(arr)
#     cdef double *out_ptr = <double *>malloc(length * sizeof(double))
#     memcpy(out_ptr, data_ptr, length*sizeof(double))

#     return out_ptr


cdef public double* decimate(
    int k,
    double *signal,
    int q
):
    """Decimate a signal

    k: number of samples in signal
    signal: pointer to the signal
    q: decimation factor

    This implementation was inspired by scipy.signal.decimate.

    """
    # signal_arr will not own the data, signal's array has to be freed
    # after return from this function
    signal_arr = PyArray_SimpleNewFromData(
        1,                      # nd
        [k],                    # dims
        np.NPY_DOUBLE,          # typenum
        <void *>signal          # data
    )

    # resampled = dsp.resample(
    #     signal_arr,
    #     len(signal_arr) // q
    # )

    b = dsp.firwin(q+1, 1./q, window='hamming')
    a = [1.]

    filtered = dsp.filtfilt(
        b=b,
        a=a,
        x=signal_arr
    )

    resampled = filtered[::q]

    if not resampled.flags['C_CONTIGUOUS']:
        resampled = resampled.copy(order='C')

    # Copy data to output array
    cdef double *resampled_ptr = <double *>np.PyArray_DATA(resampled)
    cdef double *out_ptr = <double *>malloc(len(resampled)*sizeof(double))
    memcpy(out_ptr, resampled_ptr, len(resampled)*sizeof(double))

    return out_ptr
