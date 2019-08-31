# lts_array #
This package contains a least trimmed squares algorithm modified for array processing written in Python3. An extensive module of helper functions is also included. These codes are referenced in
> Bishop, J.W., Fee, D., & Szuberla, C. A. L., 2019. Improved infrasound array processing with robust estimators, Geophysical Journal International, p. In prep

## Motivation ##
Infrasonic and seismic array processing often rely on the plane wave assumption. With this assumption, inter-element travel times can be regressed over station (co-)array coordinates to determine an optimal back-azimuth and velocity of impinging signals. Station errors such as clock issues, reversed polarity, and flat channels can appear as apparent deviations from the plane wave assumption - outlying travel times. Additionally, physical deviations from the plane wave assumption also appear as outlying data. This project identifies outlying travel times from infrasound (and seismic) arrays through the _least trimmed squares_ robust regression technique. Our python implementation uses the FAST_LTS algorithm of _Rousseeuw and Van Driessen (2006)_. Please see _Bishop et al. (2019)_ for processing examples at arrays from the International Monitoring System and Alaska Volcano Oberservatory.

## Installation ##
For your convenience, a conda environment file (pyarray.yml) is provided that includes the package specifications of the original build environment. Most prominantly, the build includes:
 1. Python = 3.6.5
 2. numpy = 1.13.3
 3. scipy = 1.1.0

After downloading the pyarray.yml file, we recommend creating an array processing environment with the command
`conda env create -f pyarray.yml`. 
After creating the environment, activate it with
`conda activate pyarray`
and verify that the proper environment is installed with
`conda list`.

This information on conda environments (and more) is available at [](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Example Processing ##

## Credits and References ##
If you use this code for array processing, we ask that you cite the following papers:

1. Bishop, J.W., Fee, D., & Szuberla, C. A. L., 2019. Improved infrasound array processing with robust estimators, Geophysical Journal International, p. In prep

2. Rousseeuw, P. J. & Van Driessen, K., 2006. Computing LTS regression for large data sets, Data Mining and Knowledge Discovery, 12(1), 29-45


## License ##
MIT (c) Jordan W Bishop
