cochlea3 -- Work In Progress, i.e., API will change!
========

*cochlea3* is a collection of inner ear models.  All models are easily
accessible as Python3 functions.  They take sound signal as input and
return `spike trains`_ of the auditory nerve fibers::



                           +----------+     __|______|______|____
   .-.     .-.     .-.     |          |-->  _|________|______|___
  /   \   /   \   /   \ -->| Cochlea3 |-->  ___|______|____|_____
       '-'     '-'         |          |-->  __|______|______|____
                           +----------+
            Sound                               Spike Trains
                                              (Auditory Nerve)



The package contains state-of-the-art biophysical models, which give
realistic approximation of the auditory nerve activity.

Whenever possible, the models were implemented using the original code
from their authors.  As a result, they provide the same responses as
the original models.  In most cases, it was verified by the unit
testing (see tests directory for details).

The implementation is also fast.  It is easy to generate responses of
hundreds or even thousands of auditory nerve fibers (ANFs).  For
example, one can generate responses of the whole human auditory nerve
(around 30,000 ANFs).  The models were usually tested with sounds of
up to 1 second in duration.

*cochlea3* is derived from *cochlea* but with Python 3 support and
some minor changes.

I developed *cochlea* during my PhD in the group of Werner Hemmert
(`Bio-Inspired Information Processing`_) at the TUM.

.. _`spike trains`: https://en.wikipedia.org/wiki/Spike_train
.. _`Bio-Inspired Information Processing`: https://www.ei.tum.de/en/bai/home/



Features
--------

- State of the art inner ear models accessible from Python 3.
- Contains full biophysical inner ear models: sound in, spikes out.
- Fast; can generate thousands of spike trains.
- Can be used with with neuron simulation software such as NEURON_ or Brian_.

.. _NEURON: http://www.neuron.yale.edu/neuron/
.. _Brian: http://briansimulator.org/



Implemented Models
------------------

- Holmberg, M. (2007). Speech Encoding in the Human Auditory
  Periphery: Modeling and Quantitative Assessment by Means of
  Automatic Speech Recognition. PhD thesis, Technical University
  Darmstadt.
- Zilany, M. S., Bruce, I. C., Nelson, P. C., &
  Carney, L. H. (2009). A phenomenological model of the synapse
  between the inner hair cell and auditory nerve: long-term adaptation
  with power-law dynamics. The Journal of the Acoustical Society of
  America, 126(5), 2390-2412.
- Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated
  parameters and expanded simulation options for a model of the
  auditory periphery. The Journal of the Acoustical Society of
  America, 135(1), 283-286.



Usage
-----

Initialize the modules::

  import cochlea3


Generate sound::

  fs = 100e3
  sound = wv.ramped_tone(
      fs=fs,
      freq=1000,
      duration=0.1,
      dbspl=50
  )


Run the model (responses of 200 cat HSR fibers)::

  anf_trains = cochlea.run_zilany2014(
      sound,
      fs,
      anf_num=(200,0,0),
      cf=1000,
      seed=0,
      species='cat'
  )


Plot the results::

  th.plot_raster(anf_trains)
  th.show()



More examples are available in examples_ directory.

.. _examples: ./examples



Installation
------------

::

  pip3 install cochlea3

Check INSTALL.rst_ for more details.

.. _INSTALL.rst: ./INSTALL.rst




Spike Train Format
------------------

All models return spike trains in a common format.  The format is
based on standard Python data structures (list, dict) and Numpy
arrays.  It contains of a list of dicts where each dict contains
standard keys: 'type', 'cf', 'offset', 'duration', 'spikes'.



Spike train data format is based on a standard DataFrame_ format from
the excellent pandas_ library.  Spike trains and their meta data are
stored in DataFrame_, where each row corresponds to a single neuron:

=====  ========  ====  ====  =================================================
index  duration  type    cf                                             spikes
=====  ========  ====  ====  =================================================
0          0.15   hsr  8000  [0.00243, 0.00414, 0.00715, 0.01089, 0.01358, ...
1          0.15   hsr  8000  [0.00325, 0.01234, 0.0203, 0.02295, 0.0268, 0....
2          0.15   hsr  8000  [0.00277, 0.00594, 0.01104, 0.01387, 0.0234, 0...
3          0.15   hsr  8000  [0.00311, 0.00563, 0.00971, 0.0133, 0.0177, 0....
4          0.15   hsr  8000  [0.00283, 0.00469, 0.00929, 0.01099, 0.01779, ...
5          0.15   hsr  8000  [0.00352, 0.00781, 0.01138, 0.02166, 0.02575, ...
6          0.15   hsr  8000  [0.00395, 0.00651, 0.00984, 0.0157, 0.02209, 0...
7          0.15   hsr  8000  [0.00385, 0.009, 0.01537, 0.02114, 0.02377, 0....
=====  ========  ====  ====  =================================================

The column 'spikes' is the most important and stores an array with
spike times (time stamps) in seconds for every action potential.  The
column 'duration' is the duration of the sound.  The column 'cf' is
the characteristic frequency (CF) of the fiber.  The column 'type'
tells us what auditory nerve fiber generated the spike train.  'hsr'
is for high-spontaneous rate fiber, 'msr' and 'lsr' for medium- and
low-spontaneous rate fibers.

Advantages of the format:

- easy addition of new meta data,
- efficient grouping and filtering of trains using DataFrame_
  functionality,
- export to MATLAB struct array through mat files::

    scipy.io.savemat(
        "spikes.mat",
        {'spike_trains': spike_trains.to_records()}
    )

The library thorns_ has more information and functions to manipulate
spike trains.


.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
.. _pandas: http://pandas.pydata.org/
.. _thorns: https://github.com/mrkrd/thorns



Contribute & Support
--------------------

- Open tasks: TODO.org_ (best viewed in Emacs org-mode)
- Issue Tracker: https://github.com/mrkrd/cochlea/issues
- Source Code: https://github.com/mrkrd/cochlea

.. _TODO.org: TODO.org



Similar Projects
----------------

- `Carney Lab`_
- `Matlab Auditory Periphery`_
- DSAM_
- `Brian Hears`_
- `The Auditory Modeling Toolbox`_

.. _`Carney Lab`: http://www.urmc.rochester.edu/labs/Carney-Lab/publications/auditory-models.cfm
.. _DSAM: http://dsam.org.uk/
.. _`Matlab Auditory Periphery`: http://www.essexpsychology.macmate.me/HearingLab/modelling.html
.. _`Brian Hears`: http://www.briansimulator.org/docs/hears.html
.. _`The Auditory Modeling Toolbox`: http://amtoolbox.sourceforge.net/



Citing
------

Rudnicki M., Schoppe O., Isik M., Völk F. and
Hemmert W. (2015). *Modeling auditory coding: from sound to spikes*.
Cell and Tissue Research, Springer Nature, 361, pp. 159—175.
doi:10.1007/s00441-015-2202-z
https://link.springer.com/article/10.1007/s00441-015-2202-z


BibTeX entry::

  @Article{Rudnicki2015,
    author    = {Marek Rudnicki and Oliver Schoppe and Michael Isik and Florian Völk and Werner Hemmert},
    title     = {Modeling auditory coding: from sound to spikes},
    journal   = {Cell and Tissue Research},
    year      = {2015},
    volume    = {361},
    number    = {1},
    pages     = {159--175},
    month     = {jun},
    doi       = {10.1007/s00441-015-2202-z},
    publisher = {Springer Nature},
  }


Do not forget to cite the original authors of the models as listed in
Implemented Models.



Acknowledgments
---------------

We would like to thank Muhammad S.A. Zilany, Ian C. Bruce and
Laurel H. Carney for developing inner ear models and allowing us to
use their code in *cochlea*.

Thanks goes to Marcus Holmberg, who developed the traveling wave based
model.  His work was supported by the General Federal Ministry of
Education and Research within the Munich Bernstein Center for
Computational Neuroscience (reference No. 01GQ0441, 01GQ0443 and
01GQ1004B).

We are grateful to Ray Meddis for support with the Matlab Auditory
Periphery model.

And last, but not least, I would like to thank Werner Hemmert for
supervising my PhD.  The thesis entitled *Computer models of
acoustical and electrical stimulation of neurons in the auditory
system* can be found at https://mediatum.ub.tum.de/1445042

This work was supported by the General Federal Ministry of Education
and Research within the Munich Bernstein Center for Computational
Neuroscience (reference No. 01GQ0441 and 01GQ1004B) and the German
Research Foundation Foundation's Priority Program PP 1608 *Ultrafast
and temporally precise information processing: Normal and
dysfunctional hearing*.


License
-------

The project is licensed under the GNU General Public License v3 or
later (GPLv3+).
