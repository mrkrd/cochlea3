from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import itertools

from . import traveling_waves as tw
from . traveling_waves import real_freq_map, get_nearest_cf

from . holmberg2007_vesicles import run_holmberg2007_vesicles


def run_holmberg2007(
        sound,
        fs,
        anf_num,
        seed,
        cf=None,
        syn_mode='probability'
):
    """Run the inner ear model by [Holmberg2007]_.  It simulates the
    traveling wave on the basilar membrane, inner hair cell, synapses
    and generates auditory nerve spikes.  The model takes sound signal
    as input and outputs auditory nerve spike trains.


    Parameters
    ----------
    sound : array_like
        Input sound signal.
    fs : float
        Sampling frequency of the sound.
    anf_num : tuple
        Number of auditory nerve fibers per channel (HSR#, MSR#, LSR#).
    seed : int
        Random seed.
    cf : float or array_like or None, optional
        Characteristic frequencies.  If `None`, then calculate all 100
        predefined CFs.  CFs must be a subset of `real_freq_map`.
    syn_mode : {'probability', 'quantal'}, optional
        Output mode of the IHC synapse.


    Returns
    -------
    spike_trains
        Auditory nerve spike trains.


    References
    ----------
    If you are using results of this or modified version of the model
    in your research, please cite [Holmberg2007]_.

    .. [Holmberg2007] Holmberg, M. (2007). Speech Encoding in the
       Human Auditory Periphery: Modeling and Quantitative Assessment
       by Means of Automatic Speech Recognition. PhD thesis, Technical
       University Darmstadt.

    """

    assert np.max(np.abs(sound)) < 1000, "Signal should be given in Pa"
    assert sound.ndim == 1
    assert fs == 48e3

    np.random.seed(seed)

    if cf is None:
        cfs = tw.real_freq_map
    elif np.isscalar(cf):
        cfs = [cf]
    else:
        cfs = cf


    assert set(cfs) <= set(tw.real_freq_map), set(cfs) - set(tw.real_freq_map)


    duration = len(sound) / fs



    ### Outer ear filter
    sound_oe = tw.run_outer_ear_filter(sound, fs)

    ### Middle ear filter
    sound_me = tw.run_middle_ear_filter(sound_oe, fs)

    ### Scaling
    sound_scaled = sound_me * tw.scaling_factor

    ### Basilar membrane
    xbm = tw.run_bm_wave(sound_scaled, fs)


    ihcrp = {}
    for cf in cfs:
        ### Amplification
        lcr4 = tw.run_lcr4(xbm[cf], fs, cf)

        ### Delay correction (1/cf)
        sec = np.where(tw.real_freq_map == cf)[0][0]
        lcr4_rolled = np.roll(
            lcr4,
            -int(np.round(tw.delay_time[99-sec]*fs))
        )

        ### IHCRP
        ihcrp[cf] = tw.run_ihcrp(lcr4_rolled, fs, cf)



    anf_types = np.repeat(['hsr', 'msr', 'lsr'], anf_num)


    ihc_meddis2000_pars = {
        'hsr': {                # H2 fiber from Sumner et al. (2002)
            'gamma_ca': 130,
            'beta_ca': 400,
            'tau_m': 1e-4,
            'gmax_ca': 4.5e-9,
            'e_ca': 0.066,
            'tau_ca': 1e-4,
            'perm_ca0': 0,
            'perm_z': 2e32,
            'power_ca': 3,
            'replenish_rate_y': 10,
            'loss_rate_l': 2580,
            'recovery_rate_r': 6580,
            'reprocess_rate_x': 66.3,
            'max_free_pool': 8,
        },
        'msr': {
            'gamma_ca': 130,
            'beta_ca': 400,
            'tau_m': 1e-4,
            'gmax_ca': 4.25e-9,
            'e_ca': 0.066,
            'tau_ca': 1e-4,
            'perm_ca0': 2.5e-11,
            'perm_z': 2e32,
            'power_ca': 3,
            'replenish_rate_y': 10,
            'loss_rate_l': 2580,
            'recovery_rate_r': 6580,
            'reprocess_rate_x': 66.3,
            'max_free_pool': 9,
        },
        'lsr': {
            'gamma_ca': 130,
            'beta_ca': 400,
            'tau_m': 1e-4,
            'gmax_ca': 2.75e-9,
            'e_ca': 0.066,
            'tau_ca': 1e-4,
            'perm_ca0': 4.2e-11,
            'perm_z': 2e32,
            'power_ca': 3,
            'replenish_rate_y': 10,
            'loss_rate_l': 2580,
            'recovery_rate_r': 6580,
            'reprocess_rate_x': 66.3,
            'max_free_pool': 6,
        }
    }

    psps = {}
    trains = []
    for cf,anf_type in itertools.product(ihcrp.keys(),anf_types):

        if (syn_mode == 'quantal') or ((cf,anf_type) not in psps):
            ### IHC and Synapse
            psp = tw.run_ihc_meddis2000(
                ihcrp=ihcrp[cf],
                fs=fs,
                syn_mode=syn_mode,
                **ihc_meddis2000_pars[anf_type]
            )
            psps[cf,anf_type] = psp

        elif (syn_mode == 'probability') and ((cf,anf_type) in psps):
            psp = psps[cf,anf_type]

        else:
            raise RuntimeError


        ### Spike generator (pars from Sumner et al. 2002)
        spikes = tw.run_an_sg_carney_holmberg2007(
            psp=psp,
            fs=fs,
            c0=0.5,
            c1=0.,
            s0=0.8e-3,
            s1=12.5e-3,
            refractory_period=0.75e-3,
        )

        trains.append({
            'spikes': spikes,
            'duration': duration,
            'cf': cf,
            'type': anf_type
        })


    spike_trains = pd.DataFrame(trains)
    return spike_trains
