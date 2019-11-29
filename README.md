# lts_array #
This package contains a least trimmed squares algorithm modified for array processing written in Python3. An extensive module of helper functions is also included. These codes are referenced in
> Bishop, J.W., Fee, D., & Szuberla, C. A. L., (in review). Improved infrasound array processing with robust estimators, Geophysical Journal International

## Motivation ##
Infrasonic and seismic array processing often relies on the plane wave assumption. With this assumption, inter-element travel times can be regressed over station (co-)array coordinates to determine an optimal back-azimuth and velocity of impinging signals. Station errors such as clock issues, reversed polarity, and flat channels can appear as apparent deviations from the plane wave assumption - outlying travel times. Additionally, physical deviations from the plane wave assumption also appear as outliers. This project identifies outlying travel times from infrasound (and seismic) arrays through the _least trimmed squares_ robust regression technique. Our python implementation uses the FAST_LTS algorithm of _Rousseeuw and Van Driessen (2006)_. Please see _Bishop et al. (in review)_ for processing examples at arrays from the International Monitoring System and Alaska Volcano Observatory.

## Installation ##
We recommend using conda and suggest creating a new conda environment to use this repository:
```
conda create -n uafinfra python=3 obspy
```
Information on conda environments (and more) is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

After setting up the conda environment, [install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) the package by running the terminal commands:
```
conda activate uafinfra
git clone https://github.com/uafgeotools/lts_array
cd lts_array
pip install -e .
```

## Dependencies ##
* [Python3](https://docs.python.org/3/)
* [ObsPy](http://docs.obspy.org/)

and their dependencies.

## Usage ##
To access the functions in this package, use the following line (for example):
```
>> python
import lts_array as lts_array
```

## Example Processing ##
See the included `Example_Processing.py`.

## References and Credits ##
If you use this code for array processing, we ask that you cite the following papers:

1. Bishop, J.W., Fee, D., & Szuberla, C. A. L., (in review). Improved infrasound array processing with robust estimators, Geophysical Journal International

2. Rousseeuw, P. J. & Van Driessen, K., 2006. Computing LTS regression for large data sets, Data Mining and Knowledge Discovery, 12(1), 29-45

## License and Authors ##
MIT (c) Jordan W Bishop, David Fee, Curt Szuberla
