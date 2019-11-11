#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import lstsq

from copy import deepcopy

import flts_helper_array as fltsh

def fast_lts_array(X, y, ALPHA): # noqa
    r''' A FAST-LTS code modified for array processing.

    @author: Jordan W. Bishop

    This code is based off the FAST-LTS algorithm:
    Rousseeuw, Peter J. and Katrien Van Driessen (2006). "Data Mining and
        Knowledge Discovery". In: Springer Science + Business Media, Inc.
        Chap. 12: Computing LTS Regression for Large Data Sets, pp. 29-45

    ASSUMPTIONS:
    This algorithm is designed for 2D velocity - back-azimuth array processing,
    so some sophistication is left out from what would be a more
    generalized version. As such, we are assuming a plane wave model,
    so (from the physics) the intercept is assumed to be zero. As a result,
    "intercept adjustment" protocols of the full FAST-LTS algorithm are
    left out of processing. We will always assume that we are dealing with
    a relatively small group of data n <= 100.

    Args:
        1. X - [array] The design matrix (co-array coordinates).
        2. y - [array] The vector of response variables
            (inter-element travel times).
        3. alpha - [float] The subset percentage to take - must be in
            the range of [0.5, 1.0]. A good default alpha is 0.75.

    Returns:
        1. Res - [dictionary] A dictionary of output parameters:
            a. bazimuth - [float] Back-azimuth in degrees from North.
            b. velocity - [float] Velocity.
            c. coefficients - [array] The x and y components of
                the slowness vector [sx, sy].
            d. flagged - [array] An array of the binary weights
                used the final weighted least squares fit.
            e. fitted - [array] The value of the best-fit plane at
                the co-array coordinates.
            f. residuals - [array] The residuals between the "fitted"
                values and the "y" values.
            g. scale - [float] The standard deviation of the best fitting
                subset multiplied by small sample correction factors.
            h. rsquared - [float] The R**2 value of the regression fit.
            j. X - [array] The input co-array coordinate array.
            k. y - [array] The input inter-element travel times.

    '''

    fast_lts_array.__version__ = '1.1'

    # Default values
    csteps1 = 4
    csteps2 = 100

    seed = 0
    intercept = 0
    ntrial = 500

    Xvar = deepcopy(X) # noqa
    yvar = deepcopy(y)

    # Initial error checking for inputs
    n, p = np.shape(Xvar)
    dimy1, dimy2 = np.shape(yvar)

    if n != dimy1:
        raise IndexError('Arrays are incompatible! X[0] != y[0]!')
    if dimy2 != 1:
        raise IndexError('y is not a column vector!')
    if n <= 2*p:
        raise ValueError('Bad assumption, n ! > 2p')

    bestobj = np.inf

    # Checking the rank of X
    rk = np.linalg.matrix_rank(Xvar)
    if rk < p:
        print('X is singular!!')

    # Assigning the subset size
    h = fltsh.hcalc(ALPHA, n, p)

    if p < 5:
        eps = 1e-12
    elif p <= 18:
        eps = 1e-14
    else:
        eps = 1e-16

    # Standardizing the data as recommended in Rousseeuw and Leroy (1987)
    xorig = deepcopy(Xvar)
    yorig = deepcopy(yvar)
    data = np.concatenate((Xvar, yvar), axis=1)

    datamad = 1.4826*np.median(np.abs(data), axis=0)
    for ii in range(0, p+1):
        if np.abs(datamad[ii]) <= eps:
            datamad[ii] = np.sum(np.abs(data[:, ii]))
            datamad[ii] = 1.2533 * (datamad[ii]/n)
            if np.abs(datamad[ii]) <= eps:
                print('ERROR: The MAD of some variable is zero!')
    for ii in range(0, p):
        Xvar[:, ii] = Xvar[:, ii]/datamad[ii]
    yvar[:, 0] = yvar[:, 0]/datamad[p]

    # Starting the algorithm; Pre-allocating best objective
    # functions and best coefficients
    part, adjh, nsamp = 0, h, ntrial
    csteps = csteps1
    tottimes, final = 0, 0

    z = np.zeros((p, 1))
    bobj = np.zeros(10, )
    bobj.fill(np.inf)
    bcoeff = np.zeros((p, 10))
    bcoeff.fill(np.nan)
    coeffs = np.tile(np.nan, (p, 1))

    # Checking to see if ALPHA == 1.00.
    # If so, perform an ordinary least squares fit and exit.
    if ALPHA == 1.00:
        lst_sq_estimate = fltsh.least_squares_fit(Xvar,
                                                  yvar, datamad, xorig, yorig)
        return lst_sq_estimate

    # Starting the search through subsets
    while final != 2:
        if final:
            nsamp = 10
            adjh = h
            if (n*p <= 1e5):
                csteps = csteps2
            elif (n*p <= 1e6):
                csteps = 10 - (np.ceil(n*p/1e5) - 2)
            else:
                csteps = 1
            if n > 5000:
                nsamp = 1
        for ii in range(0, nsamp):
            tottimes += 1
            prevobj = 0
            if final:
                if np.isfinite(bobj[ii]):
                    z = deepcopy(bcoeff[:, ii])
                    z = np.reshape(z, (len(z), 1))
                else:
                    # print('BREAK! Line 168')
                    break
            else:
                z[0, 0] = np.inf
                while z[0, 0] == np.inf:
                    index, seed = fltsh.randomset(n, p, seed)
                    index -= 1
                    q, r = np.linalg.qr(Xvar[index, :])
                    qt = q.conj().T @ yvar[index]
                    z = lstsq(r, qt)[0]

            if np.isfinite(z[0]):
                residu = yvar - Xvar@z
                # Perform C-steps
                for jj in range(0, csteps):
                    tottimes += 1
                    sortind = np.argsort(
                        np.abs(residu), kind='mergesort', axis=0)
                    sortind = np.reshape(sortind, (len(sortind), ))
                    obs_in_set = np.sort(
                        sortind[0:adjh], kind='mergesort', axis=0)
                    q, r = np.linalg.qr(Xvar[obs_in_set, :])
                    qt = q.conj().T @ yvar[obs_in_set]
                    z = lstsq(r, qt)[0]
                    residu = yvar - Xvar@z
                    sor = np.sort(np.abs(residu), kind='mergesort', axis=0)
                    # New objective function
                    obj = np.sum(sor[0:adjh:1]**2)
                    if (jj >= 1) and (obj == prevobj):
                        break
                    prevobj = deepcopy(obj)

                if (not final):
                    if obj < np.max(bobj):
                        # Save the best objective function values
                        bcoeff, bobj = fltsh.insertion(
                            bcoeff, bobj, z, obj)
                if final and (obj < bestobj):
                    bestobj = deepcopy(obj)
                    coeffs = deepcopy(z)

        if (not part) and (not final):
            final = 1
        else:
            final = 2

    # Post-processing
    if p <= 1:
        coeffs[0] = coeffs[0] * datamad[p]/datamad[0]
    else:
        for ii in range(0, p):
            coeffs[ii] = coeffs[ii] * datamad[p] / datamad[ii]
        coeffs[p-1] = coeffs[p-1] * datamad[p-1] / datamad[p-1]
    bestobj = bestobj*(datamad[p]**2)
    xvar = deepcopy(xorig)
    yvar = deepcopy(yorig)

    # Saving the raw data - the intermediate results
    Raw = {} # noqa
    Raw['coefficients'] = coeffs
    Raw['objective'] = bestobj

    # Preparing for the final results
    Res = {} # noqa

    coeffs2 = np.reshape(coeffs, (len(coeffs), 1))
    fitted = xvar @ coeffs2
    Raw['fitted'] = fitted
    residuals = yvar - fitted
    Raw['residuals'] = residuals
    sor = np.sort(residuals**2, kind='mergesort', axis=0)
    factor = fltsh.rawcorfactorlts(p, intercept, n, ALPHA)
    factor = factor * fltsh.rawconsfactorlts(h, n)
    sh0 = np.sqrt((1/h)*np.sum(sor[0:h]))
    s0 = sh0 * factor

    if np.abs(s0) < 1e-7:
        weights = (np.abs(residuals) < 1e-7)*1
        weights = np.reshape(weights, (len(weights), ))
        Raw['wt'] = weights
        Raw['scale'] = 0
        Res['scale'] = 0
        Res['coefficients'] = deepcopy(Raw['coefficients'])
        Raw['objective'] = 0
    else:
        Raw['scale'] = s0
        quantile = fltsh.qnorm(0.9875)
        weights = (np.abs(residuals/s0) <= quantile)
        weights = np.reshape(weights, (len(weights), ))
        weights2 = weights*1
        Raw['weights'] = weights2
        # Now perform the least squares fit with
        # only data points with weight = 1.
        q, r = np.linalg.qr(xvar[weights, :])
        qt = q.conj().T @ yvar[weights]
        zfinal = lstsq(r, qt)[0]
        Res['coefficients'] = deepcopy(zfinal)
        fitted = xvar @ zfinal
        residuals = yvar - fitted
        residuals = np.reshape(residuals, (len(residuals), ))
        Res['scale'] = np.sqrt(np.sum(
            np.multiply(weights2, residuals)**2)/(np.sum(weights2) - 1))
        factor = fltsh.rewcorfactorlts(p, intercept, n, ALPHA)
        factor *= fltsh.rewconsfactorlts(weights, n, p)
        Res['scale'] *= factor
        weights = np.abs(residuals/Res['scale']) <= 2.5
        weights = np.reshape(weights, (len(weights), )) * 1

    Res['flagged'] = deepcopy(weights)
    sor = np.sort(residuals**2, kind='mergesort', axis=0)
    s1 = np.sum(sor[0:h])
    sor2 = np.sort(yvar**2, kind='mergesort')
    sh = np.sum(sor2[0:h])
    rsquared = 1 - (s1/sh)
    if rsquared > 1:
        rsquared = 1
    elif rsquared < 0:
        rsquared = 0
    Res['rsquared'] = deepcopy(rsquared)
    Res['residuals'] = deepcopy(residuals)
    Res['sigma_tau'] = np.nan
    if np.abs(s0) < 1e-7:
        print('An exact fit was found!')
    Res['fitted'] = deepcopy(fitted)
    Res['X'] = deepcopy(xorig)
    Res['y'] = deepcopy(yorig)

    s_p = Res['coefficients']
    vel = 1/np.linalg.norm(s_p, 2)
    # this converts az from mathematical CCW from E to geographical CW from N
    # arctan(sx/sy)
    az = (np.arctan2(s_p[0], s_p[1])*180/np.pi-360) % 360
    Res['velocity'] = vel
    Res['bazimuth'] = np.asscalar(az)

    return Res
