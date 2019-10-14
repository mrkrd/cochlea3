#!/usr/bin/env python3

"""This example illustrates how to run Zilany et al. (2014) model with
the rate (not spikes) as output.

This functionality is provided for “as is” without much support.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp

import cochlea3


def main():

    fs = 100e3

    # Make sound
    t = np.arange(0, 0.1, 1/fs)
    s = dsp.chirp(t, 80, t[-1], 20000)
    s = cochlea3.set_dbspl(s, 50)
    s = np.concatenate((s, np.zeros(int(10e-3 * fs))))

    # Run model
    rates = cochlea3.run_zilany2014_rate(
        s,
        fs,
        anf_types=['msr'],
        cf=(125, 20000, 3),
        powerlaw='approximate',
        species='human'
    )

    # Plot rates
    fig, ax = plt.subplots()
    for rate in rates:
        ax.plot(rate['rate'])
    plt.show()


if __name__ == "__main__":
    main()
