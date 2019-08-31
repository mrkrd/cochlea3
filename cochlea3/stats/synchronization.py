"""Synchronization of inner ear models.

"""
from __future__ import division, absolute_import, print_function
from __future__ import unicode_literals

import numpy as np

import thorns as th
import thorns.waves as wv


def calc_synchronization(
        model,
        cfs=None,
        dbspls=None,
        model_pars=None,
        map_backend=None,
):
    """Calculate vector strength of an inner ear model.

    """
    if model_pars is None:
        model_pars = {}

    if cfs is None:
        cfs = np.logspace(np.log10(125), np.log10(16e3), 16)

    if dbspls is None:
        dbspls = [20, 40, 60]

    space = {
        'dbspl': dbspls,
        'cf': cfs,
    }

    kwargs = {
        'model': model,
        'model_pars': model_pars,
    }

    all_vss = th.util.map(
        _run_model,
        space,
        kwargs=kwargs,
        backend=map_backend,
    )

    vss = all_vss.reset_index()

    best_vss = vss.groupby('cf').max().drop('dbspl', axis=1)

    return best_vss


def _run_model(model, dbspl, cf, model_pars):

    duration = 100e-3
    onset = 10e-3

    fs = model_pars.setdefault('fs', 100e3)
    model_pars.setdefault('anf_num', (250, 250, 250))
    model_pars.setdefault('seed', 0)

    sound = wv.ramped_tone(
        fs=fs,
        freq=cf,
        duration=duration,
        pad=0,
        dbspl=dbspl
    )

    anf = model(
        sound=sound,
        cf=cf,
        **model_pars
    )

    # th.plot_raster(anf)
    # th.show()

    # We want to make sure the the output CF is equal to the desired
    # CF.
    real_cf, = np.unique(anf['cf'])
    assert real_cf == cf

    vss = {}
    for typ, group in anf.groupby('type'):
        trimmed = th.trim(group, onset, None)
        vs = th.vector_strength(trimmed, cf)
        vss[typ] = vs

    return vss
