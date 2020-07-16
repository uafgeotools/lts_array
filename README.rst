lts_array
=========

This package contains a least trimmed squares algorithm modified for
geophysical array processing and written in Python3. An extensive module
of helper functions is also included. These codes are referenced in >
Bishop, J.W., Fee, D., & Szuberla, C. A. L., (2020). Improved infrasound
array processing with robust estimators, Geophys. J. Int., 221
p. 2058-2074 doi: https://doi.org/10.1093/gji/ggaa110.

A broader set of geophysical array processing codes are available
`here <https://github.com/uafgeotools/array_processing>`__, which
utilizes this package as the default (and preferred) array processing
algorithm.

Motivation
----------

Infrasonic and seismic array processing often relies on the plane wave
assumption. With this assumption, inter-element travel times can be
regressed over station (co-)array coordinates to determine an optimal
back-azimuth and velocity for waves crossing the array. Station errors
such as digitizer timing issues, reversed polarity, and flat channels
can manifest as apparent deviations from the plane wave assumption as
travel time outliers. Additionally, physical deviations from the plane
wave assumption also appear as travel time outliers. This project
identifies these outliers from infrasound (and seismic) arrays through
the *least trimmed squares* robust regression technique. Our python
implementation uses the FAST_LTS algorithm of *Rousseeuw and Van
Driessen (2006)*. Please see *Bishop et al. (2020)* for processing
examples at arrays from the International Monitoring System and Alaska
Volcano Observatory.

Installation
------------

We recommend using conda and creating a new conda environment such as:

::

   conda create -n py_array -c conda-forge python=3 obspy

Information on conda environments (and more) is available
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

After setting up the conda environment,
`install <https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs>`__
the package by running the terminal commands:

::

   conda activate py_array
   git clone https://github.com/uafgeotools/lts_array
   cd lts_array
   pip install -e .

The final command installs the package in “editable” mode, which means
that you can update it with a simple ``git pull`` in your local
repository. This install command only needs to be run once.

Dependencies
------------

-  `Python3 <https://docs.python.org/3/>`__
-  `ObsPy <http://docs.obspy.org/>`__

and their dependencies.

Usage
-----

To access the functions in this package, use the following line (for
example):

::

   >> python
   import lts_array as lts_array

Example Processing
------------------

See the included ``Example_Processing.py``.

References and Credits
----------------------

If you use this code for array processing, we ask that you cite the
following papers:

1. Bishop, J.W., Fee, D., & Szuberla, C. A. L., (2020). Improved
   infrasound array processing with robust estimators, Geophys. J. Int.,
   Vol(num) p. ###-### doi: https://doi.org/10.1093/gji/ggaa110

2. Rousseeuw, P. J. & Van Driessen, K., 2006. Computing LTS regression
   for large data sets, Data Mining and Knowledge Discovery, 12(1),
   29-45 doi: https://doi.org/10.1007/s10618-005-0024-4

License
-------

MIT (c)

Authors and Contributors
------------------------

| Jordan W Bishop
| David Fee
| Curt Szuberla
| Liam Toney

Acknowledgements and Distribution Statement
-------------------------------------------

This work was made possible through support provided by the Defense
Threat Reduction Agency Nuclear Arms Control Technology program under
contract HDTRA1-17-C-0031. Distribution Statement A: Approved for public
release; distribution is unlimited.
