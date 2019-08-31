#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Process with robust estimators.

@author: Jordan W Bishop

Date Last Modified: 8/29/19

Args:
  1. data - [array] - Array of time series for cross correlations
  2. fs - [scalar] - sampling rate
  3. rij - [array] - array coordinates
  4. alpha - [scalar] - fraction of data for LTS subsetting [0.5, 1.0]

Returns:
  1. fltsbaz - [scalar] - the back-azimuth in degrees from North as
      determined by the least trimmed squares fit.
  2. fltsvel - [scalar] - the velocity determined by the least trimmed
      squares fit.
  3. flagged - [array] - the binary (0 or 1) weights assigned
    to station pairs in the final weighted least squares fit.
    Stations with a final weight of "0" are flagged as outlying
    by the algorithm.
  4. ccmax - [array] - The cross correlation maxima used to
      determine inter-element travel times.
  5. idx - [array] - station pairs.
  6. lts_estimate [dictionary] - a dictionary with the following keys
      a) bazimuth - [scalar] - fltsbaz
      b) velocity - [scalar] - fltsvel
      c) coefficients - [array] - the x and y components of
        the slowness vector [sx, sy].
      d) flagged - [array] - the binary (0 or 1) weights assigned
        to station pairs in the final weighted least squares fit.
        Stations with a final weight of "0" are flagged as outlying
        by the algorithm.
      e) fitted - [array] - the value of the best-fit plane at
        the co-array coordinates.
      f) residuals - [N x 1 array] - the residuals between the
        "fitted" values and the "y" values.
      g) scale - [scalar] - the scale value used to determine the LTS weights.
      h) rsquared - [scalar] - the R**2 value of the regression fit.
      j) X - [array] - the input co-array coordinate array.
      k) y - [array] - the input inter-element travel times.

"""


def ltsva(data, rij, fs, alpha):

    import numpy as np
    from fast_lts_array import fastlts
    from flts_helper_array import getcctimevec

    # Cross-correlate to determine inter-element time delays
    tdelay, xij, ccmax, idx = getcctimevec(data, rij, fs)
    tdelay = np.reshape(tdelay, (len(tdelay), 1))
    xij = xij.T

    # Apply the FAST-LTS algorithm
    lts_estimate = fastlts(xij, tdelay, alpha)

    fltsbaz = lts_estimate['bazimuth']
    fltsvel = lts_estimate['velocity']
    flagged = lts_estimate['flagged']

    return fltsbaz, fltsvel, flagged, ccmax, idx, lts_estimate
