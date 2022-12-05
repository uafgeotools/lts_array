lts_array
=========

Introduction
-------------------

This package contains a least trimmed squares algorithm written in Python3 and modified for geophysical array processing. An extensive collection
of helper functions is also included. These codes are referenced in

Bishop, J.W., Fee, D., & Szuberla, C. A. L., (2020). *Improved
infrasound array processing with robust estimators*, Geophys. J. Int.,
221(3) p. 2058-2074 doi: https://doi.org/10.1093/gji/ggaa110

A broader set of geophysical array processing codes are available
`here <https://github.com/uafgeotools/array_processing>`__, which
utilizes this package as the default (and preferred) array processing
algorithm.

Motivation
-----------------

Infrasonic and seismic array processing often relies on the plane wave
assumption. With this assumption, inter-element travel times can be
regressed over station (co-)array coordinates to determine an optimal
back-azimuth and velocity for waves crossing the array. Station errors
such as digitizer timing issues, reversed polarity, and flat channels
can manifest as apparent deviations from the plane wave assumption as
travel time outliers. Additionally, physical deviations from the plane
wave assumption also appear as travel time outliers. This method
identifies these outliers from infrasound (and seismic) arrays through
the *least trimmed squares* robust regression technique. Our python
implementation uses the FAST_LTS algorithm of *Rousseeuw and Van
Driessen (2006)*. Uncertainty estimates are calculated using the slowness ellipse method of *Szuberla and Olson (2004)*. Please see *Bishop et al. (2020)* for processing examples at arrays from the International Monitoring System and Alaska Volcano Observatory.

Installation and Usage
------------------------------------

See the README for installation and usage instructions.


References and Credits
----------------------

If you use this code for array processing, we ask that you cite the
following papers:

1. Bishop, J.W., Fee, D., & Szuberla, C. A. L., (2020). Improved infrasound array processing with robust estimators, Geophys. J. Int., 221(3) p. 2058-2074 doi: https://doi.org/10.1093/gji/ggaa110
2. Rousseeuw, P. J. & Van Driessen, K., 2006. Computing LTS regression for large data sets, Data Mining and Knowledge Discovery, 12(1), 29-45 doi: https://doi.org/10.1007/s10618-005-0024-43.
3. Szuberla, C.A.L. & Olson, J.V., 2004. Uncertainties associated with parameter estimation in atmospheric infrasound arrays, J. Acoust. Soc. Am., 115(1), 253–258. doi: https://doi.org/10.1121/1.1635407


License
-------

MIT (c)


Acknowledgements and Distribution Statement
-------------------------------------------

This work was made possible through support provided by the Defense
Threat Reduction Agency Nuclear Arms Control Technology program under
contract HDTRA1-17-C-0031. Distribution Statement A: Approved for public
release; distribution is unlimited.
