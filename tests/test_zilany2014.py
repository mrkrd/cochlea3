from numpy.testing import assert_almost_equal
import scipy.io

import os
from os.path import join

from cochlea3.zilany2014.c_wrapper import (
    run_ihc,
    run_synapse,
)


# from cochlea3.zilany2014._util import ffGn


DATADIR = join(os.path.dirname(__file__), 'data_zilany2014')


def test_run_ihc():

    m = scipy.io.loadmat(
        join(DATADIR, 'data_zilany2014.mat'),
        squeeze_me=True
    )
    fs = float(m['fs'])
    cf = float(m['cf'])
    sound = m['sound'].astype(float)
    v_ihc_target = m['vihc']

    v_ihc = run_ihc(
        sound=sound,
        cf=cf,
        fs=fs,
        species='cat',
        c_ohc=1.,
        c_ihc=1.
    )

    assert_almost_equal(
        v_ihc,
        v_ihc_target,
        decimal=15
    )


def test_run_synapse():
    """test_synapse()

    This function has problems, because it's using matlab
    implementation of `resample' and it's difficult to exactly
    reproduce the output.

    During the generation of mat-file the `ffGn' was replaced by
    `zeros' and Matlab's `resample' by self-made downsampling function
    equivalent to python's implementation:

    function resampled = resample_fake(X, P, Q)
        b = fir1(Q, 1/Q);
        a = [1];
        filtered = filtfilt(b, a, X);
        resampled = filtered(1:Q:end);

    """
    m = scipy.io.loadmat(
        join(DATADIR, 'data_zilany2014.mat'),
        squeeze_me=True
    )
    fs = float(m['fs'])
    cf = float(m['cf'])
    v_ihc = m['vihc']
    mean_rate_target = m['meanrate']

    p_spike = run_synapse(
        v_ihc=v_ihc,
        fs=fs,
        cf=cf,
        anf_type='hsr',
        powerlaw='approximate',
        seed=None
    )
    mean_rate = p_spike / (1 + 0.75e-3*p_spike)

    assert_almost_equal(
        mean_rate,
        mean_rate_target,
        decimal=10
    )


def test_ffGn():
    m = scipy.io.loadmat(
        join(DATADIR, 'data_zilany2014.mat'),
        squeeze_me=True
    )

    r = m['random_ffGn']
    y = m['y_ffGn']

    y_actual = ffGn(
        N=16,
        tdres=0.1,
        Hinput=0.2,
        noiseType=1,
        mu=10,
        random_debug=r
    )

    assert_almost_equal(y_actual, y, decimal=13)
