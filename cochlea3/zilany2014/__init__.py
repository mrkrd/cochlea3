
import itertools
import numpy as np

from cochlea3.zilany2014.c_wrapper import (
    run_ihc,
    run_synapse,
    run_spike_generator,
)


def run_zilany2014(
        sound,
        fs,
        cfs,
        lsr,
        msr,
        hsr,
        species,
        seed,
        cohc=1,
        cihc=1,
        powerlaw='approximate',
        ffGn=False,
):
    """Run the inner ear model by [Zilany2014]_.

    This model is based on the original implementation provided by the
    authors.  The MEX specific code was replaced by Python code in C
    files.  We also compared the outputs of both implementation (see
    the `tests` directory for unit tests).


    Parameters
    ----------
    sound : array_like
        The input sound in Pa.
    fs : float
        Sampling frequency of the sound in Hz.
    anf_num : tuple
        The desired number of auditory nerve fibers per frequency
        channel (CF), (HSR#, MSR#, LSR#).  For example, (100, 75, 25)
        means that we want 100 HSR fibers, 75 MSR fibers and 25 LSR
        fibers per CF.
    cfs : array_like
        A list of center frequencies.
    species : {'cat', 'human', 'human_glasberg1990'}
        Species.
    seed : int
        Random seed for the spike generator.
    cohc : float <0-1>, optional
        Degredation of the outer hair cells.
    cihc : float <0-1>, optional
        Degredation of the inner hair cells.
    powerlaw : {'approximate', 'actual'}, optional
        Defines which power law implementation should be used.
    ffGn : bool
        Enable/disable factorial Gaussian noise.


    Returns
    -------
    spike_trains
        Auditory nerve spike trains.


    References
    ----------
    If you are using results of this or modified version of the model
    in your research, please cite [Zilany2014]_.


    .. [Zilany2014] Zilany, M. S., Bruce, I. C., & Carney,
       L. H. (2014). Updated parameters and expanded simulation
       options for a model of the auditory periphery. The Journal of
       the Acoustical Society of America, 135(1), 283-286.

    """
    assert np.max(sound) < 1000, "Signal should be given in Pa"
    assert sound.ndim == 1
    assert species in ('cat', 'human', 'human_glasberg1990')


    channel_args = [
        {
            'signal': sound,
            'cf': cf,
            'fs': fs,
            'cohc': cohc,
            'cihc': cihc,
            'anf_num': anf_num,
            'powerlaw': powerlaw,
            'species': species,
            'ffGn': ffGn,
        }
        for cf in cfs
    ]

    # Run model for each channel
    nested = map(
        _run_channel,
        channel_args
    )

    # Unpack the results
    trains = itertools.chain(*nested)
    spike_trains = list(trains)

    return spike_trains


def _run_fiber(kwargs):



def _run_channel(args):

    fs = args['fs']
    cf = args['cf']
    signal = args['signal']
    cohc = args['cohc']
    cihc = args['cihc']
    powerlaw = args['powerlaw']
    anf_num = args['anf_num']
    species = args['species']
    ffGn = args['ffGn']

    # Run BM, IHC
    vihc = _zilany2014.run_ihc(
        signal=signal,
        cf=cf,
        fs=fs,
        species=species,
        cohc=float(cohc),
        cihc=float(cihc)
    )

    duration = len(vihc) / fs
    anf_types = np.repeat(['hsr', 'msr', 'lsr'], anf_num)

    synout = {}
    trains = []
    for anf_type in anf_types:

        if (anf_type not in synout) or ffGn:
            # Run synapse
            synout[anf_type] = _zilany2014.run_synapse(
                fs=fs,
                vihc=vihc,
                cf=cf,
                anf_type=anf_type,
                powerlaw=powerlaw,
                ffGn=ffGn,      # TODO: seed
            )

        # Run spike generator
        spikes = _zilany2014.run_spike_generator(
            synout=synout[anf_type],
            fs=fs,              # TODO: random seed
        )

        trains.append({
            'spikes': spikes,
            'duration': duration,
            'offset': 0.0,
            'cf': args['cf'],
            'type': anf_type
        })

    return trains
