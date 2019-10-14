# -*- coding: utf-8 -*-

"""Innear ear models in Python.

"""

import numpy as np

from .holmberg2007 import run_holmberg2007
from .holmberg2007 import run_holmberg2007_vesicles
from .zilany2009 import run_zilany2009
# from .zilany2014 import run_zilany2014
# from .zilany2014 import run_zilany2014_rate

from .holmberg2007 import real_freq_map as freq_map_holmberg2007
from .holmberg2007 import get_nearest_cf as get_nearest_cf_holmberg2007


__version__ = "1"


def set_dbspl(signal, dbspl):
    """Rescale the signal to a new level in dB SPL.

    Parameters
    ----------
    signal : array_like
        Signal for scaling.
    dbspl : float
        Desired level of the output signal in dB SPL.

    Returns
    -------
    array_like
        Scaled version of the original signal.

    """
    p0 = 20e-6
    rms = np.sqrt(np.sum(signal**2) / signal.size)

    scalled = signal * 10**(dbspl/20) * p0 / rms

    return scalled


_greenwood_pars  = {
    'human': {'A': 165.4, 'a': 60, 'k': 0.88, 'length': 35e-3},
    'cat': {'A': 456, 'a': 84, 'k': 0.8, 'length': 25e-3},
}


def greenwood(x, species=None, A=None, a=None, k=None):
    '''Greenwood function.

    Calculates the corresponding center frequency for a place on the
    basilar membrane.
    Center frequency and place on the basilar membrane can be
    connected by using the Greenwood equation [1].
    .. math:: cf = A(10^{x \\cdot a) - k)
    where cf is the center frequency in Hz and x the place on the
    basilar membrane given from apex to base (apex = 0). The parameter
    A has the unit Hz, a has a unit of 1/m and k is unit less.

    Parameters
    ----------
    x : float or ndarray
        The position of the basilar membrane in meters. Apex = 0m
    species : str
        A string specifing the species ('human' or 'cat')
        coeffients from [1])
    A, x, k : float
        Specifying the parameters of the Greenwood equation
        instead of using the implemented species. If parameters
        are given, the species parameter is ignored.

    .. [1] Greenwood, D. D. (1990). A cochlear frequency-position
    function for several species--29 years later. The Journal of the
    Acoustical Society of America, 87(6)
    '''
    if species is not None:
        pars = _greenwood_pars[species]

        A = pars['A']
        a = pars['a']
        k = pars['k']

        if np.any(x > pars['length']):
            raise ValueError("Cochlea too long.")

    cf = A * (10**(x * a) - k)

    return cf


def greenwood_inverse(cf, species=None, A=None, a=None, k=None):
    '''Inverse Greenwood function

    Calculate the place on the basilar membrane from the corresponding
    center frequency.
    Center frequency and place on the basilar membrane can be
    connected by using the Greenwood equation [1].
    .. math:: x = \frac{log_{10}(cf / A + k)}{a}
    where cf is the center frequency in Hz and x the place on the
    basilar membrane given from apex to base (apex = 0). The parameter
    A has the unit Hz, a has a unit of 1/m and k is unit less.

    Parameters
    ----------
    cf : scalar or ndarray
        The center frequency in Hz
    species : str
        A string specifing the species ('human' or 'cat')
        coeffients from [1])
    A, x, k : float
        Specifying the parameters of the Greenwood equation
        instead of using the implemented species. If parameters
        are given, the species parameter is ignored.

    Returns
    -------
    scalar or ndarray
        The position on the basilar membrane in m where Apex = 0m

    .. [1] Greenwood, D. D. (1990). A cochlear frequency-position
    function for several species--29 years later. The Journal of the
    Acoustical Society of America, 87(6)
    '''
    if species is not None:
        pars = _greenwood_pars[species]

        A = pars['A']
        a = pars['a']
        k = pars['k']

    x = np.log10((cf / A) + k) / a

    return x
