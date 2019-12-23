import numpy as np
from scipy.special import erfinv, gammainc
from scipy.stats import gamma
from scipy.linalg import lstsq

from copy import deepcopy
from obspy.geodetics.base import calc_vincenty_inverse

import lts_array.flts_helper_array as fltsh

r''' Contains the auxilliary functions called by fast_lts_array.py

Many of these codes are Python3 translations of those found in
the MATLAB Continuous Sound and Vibration Toolbox.

'''


def hcalc(ALPHA, n, p):
    r''' Generate the h-value, the number of points to fit.

    Args:
        ALPHA (float): The decimal percentage of points
            to keep. Default is 0.75.
        n (int): The total number of points.
        p (int): The number of parameters.

    Returns:
        h (int): The number of points to fit.

    '''

    h = np.floor(2*np.floor((n + p + 1)/2)
                 - n + 2*(n - np.floor((n + p + 1)/2)) * ALPHA)

    return int(h)


def uniran(seed):
    r''' Generate a random number and a new seed.

    Args:
        1. seed - [int] A seed value.

    Returns:
        1. random [float] A pseudorandom number.
        2. seed - [float] A (new) seed value.

    '''

    seed = np.floor(seed * 5761) + 999
    quot = np.floor(seed / 65536)
    seed = np.floor(seed) - np.floor(quot * 65536)
    random = float(seed / 65536)
    return random, seed


def randomset(tot, npar, seed):
    r''' Generate an array of indices and a new seed.

    This function is called if not all (p+1) subsets out of
        n will be considered. It randomly draws a subsample of
        nel cases out of tot.

    Args:
        1. tot - [int] The total number of data points.
        2. npar - [int] The number of parameters to estimate.
        3. seed - [float] A random seed.

    Returns:
        1. randset - [array] A random set of indices for choosing subsets.
        2. seed - [float] A new random seed.

    '''

    randlist = []
    for jj in range(0, npar):
        random, seed = fltsh.uniran(seed)
        num = np.floor(random * tot) + 1
        if jj > 0:
            while num in randlist:
                random, seed = fltsh.uniran(seed)
                num = np.floor(random * tot) + 1
        randlist.append(num)

    randset = np.array(randlist, dtype=int)

    return randset, seed


def qgamma(p, a):
    r''' The gamma inverse distribution function. '''

    x = np.max((a - 1, 0.1))
    dx = 1
    eps = 7/3 - 4/3 - 1
    while np.abs(dx) > 256 * eps * np.max(np.append(x, 1)):
        dx = (fltsh.pgamma(x, a) - p) / fltsh.dgamma(x, a)
        x = x - dx
        x = x + (dx - x) / 2 * float(x < 0)

    if hasattr(x, "__len__"):
        x[x == 0] = 0
        x[x == 1] = np.inf

    return x


def pgamma(x, a):
    ''' Regularized lower incomplete gamma function. '''
    g1 = gammainc(a, x)
    return g1


def dgamma(x, a):
    ''' Probability of a gamma continuous random variable. '''
    g2 = gamma.pdf(x, a)
    return g2


def qchisq(p, a):
    ''' The Chi-squared inverse distribution function. '''
    x = 2*fltsh.qgamma(p, 0.5*a)
    return x


def insertion(bestmean, bobj, z, obj):
    r''' Keep track of the value of the objective function
        and the associated parameter vector z.

    This code could likely be re-written for more simplicty.

    Args:
        1. bestmean - [array] Array of best least squares fit values.
        2. bobj - [array] Array of lowest 10 objective function values.
        3. z - [array] Trial coefficient vector.
        4. obj - [float] Trial objective function value;
            the sum of squared residuals.

    Returns:
        1. bestmean - [array] New array of best least squares fit values.
        2. bobj - [array] New array of lowest objective function values.

    '''

    insert = 1
    equ = [x for x in range(len(bobj)) if bobj[x] == obj]

    z = np.reshape(z, (len(z), ))

    for jj in equ:
        if (bestmean[:, jj] == z).all():
            insert = 0

    if insert:
        ins = np.min([x for x in range(0, 10) if obj < bobj[x]])
        if ins == 9:
            bestmean[:, ins] = z
            bobj[ins] = obj
        else:
            ins2 = np.array(list(range(ins, 9)))
            best2 = deepcopy(bestmean[:, ins2])
            bestmean[:, ins] = z
            best1 = deepcopy(bestmean[:, range(0, ins+1)])
            if ins == 0:
                m = np.shape(bestmean)[0]
                best1 = np.reshape(best1, (m, 1))
            bestmean = np.concatenate((best1, best2), axis=1)
            bobj2 = deepcopy(bobj[ins2])
            bobj[ins] = obj
            bobj1 = deepcopy(bobj[range(0, ins+1)])
            if ins == 0:
                bobj = np.append(bobj1, bobj2)
            else:
                bobj = np.concatenate((bobj1, bobj2), axis=0)

    return bestmean, bobj


def rawcorfactorlts(p, intercept, n, alpha):
    r''' Calculate small sample correction factor.

    Calculates the correction factor (from Pison et al. 2002)
        to make the LTS solution unbiased for small n.

    Args:
        1. p - [int] The rank of X, the number of parameters to fit.
        2. intercept - [int] Logical. Are you fitting an intercept?
            Set to false for array processing.
        3. n - [int] The number of data points used in processing.
        4. alpha - [float] The percentage of data points to keep in
            the LTS, i.e. h = floor(alpha*n).

    Returns:
        1. finitefactor - [float] A correction factor to make
            the LTS solution approximately unbiased
            for small (i.e. finite n).

    '''

    if intercept == 1:
        p = p - 1
    if p == 0:
        fp_500_n = 1 - np.exp(0.262024211897096)*1/(n**0.604756680630497)
        fp_875_n = 1 - np.exp(-0.351584646688712)*1/(n**1.01646567502486)
        if (alpha >= 0.500) and (alpha <= 0.875):
            fp_alpha_n = fp_500_n + (fp_875_n - fp_500_n)/0.375*(alpha - 0.500)
            fp_alpha_n = np.sqrt(fp_alpha_n)
        if (alpha > 0.875) and (alpha < 1):
            fp_alpha_n = fp_875_n + (1 - fp_875_n)/0.125*(alpha - 0.875)
            fp_alpha_n = np.sqrt(fp_alpha_n)
    else:
        if p == 1:
            if intercept == 1:
                fp_500_n = 1
                - np.exp(0.630869217886906)*1/(n**0.650789250442946)
                fp_875_n = 1
                - np.exp(0.565065391014791)*1/(n**1.03044199012509)
            else:
                fp_500_n = 1
                - np.exp(-0.0181777452315321)*1/(n**0.697629772271099)
                fp_875_n = 1
                - np.exp(-0.310122738776431)*1/(n**1.06241615923172)

        if p > 1:
            if intercept == 1:
                # ALPHA = 0.875.
                coeffalpha875 = np.array([
                    [-0.251778730491252, -0.146660023184295],
                    [0.883966931611758, 0.86292940340761],
                    [3, 5]])
                # ALPHA = 0.500.
                coeffalpha500 = np.array([
                    [-0.487338281979106, -0.340762058011],
                    [0.405511279418594, 0.37972360544988], [3, 5]])
            else:
                # ALPHA = 0.875.
                coeffalpha875 = np.array([
                    [-0.251778730491252, -0.146660023184295],
                    [0.883966931611758, 0.86292940340761], [3, 5]])
                # ALPHA = 0.500.
                coeffalpha500 = np.array([
                    [-0.487338281979106, -0.340762058011],
                    [0.405511279418594, 0.37972360544988], [3, 5]])

            # Apply eqns (6) and (7) from Pison et al. (2002)
            y1_500 = 1 + coeffalpha500[0, 0]/np.power(p, coeffalpha500[1, 0])
            y2_500 = 1 + coeffalpha500[0, 1]/np.power(p, coeffalpha500[1, 1])
            y1_875 = 1 + coeffalpha875[0, 0]/np.power(p, coeffalpha875[1, 0])
            y2_875 = 1 + coeffalpha875[0, 1]/np.power(p, coeffalpha875[1, 1])

            # Solve for new ALPHA = 0.5 coefficients for the input p.
            y1_500 = np.log(1-y1_500)
            y2_500 = np.log(1-y2_500)
            y_500 = np.array([[y1_500], [y2_500]])
            X_500 = np.array([      # noqa
                [1, np.log(1/(coeffalpha500[2, 0]*p**2))],
                [1, np.log(1/(coeffalpha500[2, 1]*p**2))]])
            c500 = np.linalg.lstsq(X_500, y_500, rcond=-1)[0]

            # Solve for new ALPHA = 0.875 coefficients for the input p.
            y1_875 = np.log(1-y1_875)
            y2_875 = np.log(1-y2_875)
            y_875 = np.array([[y1_875], [y2_875]])
            X_875 = np.array([      # noqa
                [1, np.log(1/(coeffalpha875[2, 0]*p**2))],
                [1, np.log(1/(coeffalpha875[2, 1]*p**2))]])
            c875 = np.linalg.lstsq(X_875, y_875, rcond=-1)[0]

            # Get new correction factors for the specified n.
            fp500 = 1 - np.exp(c500[0])/np.power(n, c500[1])
            fp875 = 1 - np.exp(c875[0])/np.power(n, c875[1])

            # Linearly interpolate for the specified ALPHA.
            if (alpha >= 0.500) and (alpha <= 0.875):
                fpfinal = fp500 + ((fp875 - fp500)/0.375)*(alpha - 0.500)

            if (alpha > 0.875) and (alpha < 1):
                fpfinal = fp875 + ((1 - fp875)/0.125)*(alpha-0.875)

            finitefactor = np.asscalar(1/fpfinal)
            return finitefactor


def rawconsfactorlts(h, n):
    r''' Calculate the constant used to make the
     LTS scale estimators consistent for
     a normal distrbution.

    Args:
        1. h - [int] The number of points to fit.
        2. n - [int] The total number of data points.

    Returns:
        1. dhn - [float] The correction factor d_h,n.

    '''

    # Calculate the initial factor c_h,n.
    x = (h+n)/(2*n)
    phinv = np.sqrt(2)*erfinv(2*x-1)
    chn = 1/phinv

    # Calculate d_h,n.
    phi = (1/np.sqrt(2*np.pi))*np.exp((-1/2)*phinv**2)
    d = np.sqrt(1 - (2*n/(h*chn))*phi)
    dhn = 1/d

    return dhn


def qnorm(p, s=1, m=0):
    r''' The normal inverse distribution function. '''
    x = erfinv(2*p - 1)*np.sqrt(2)*s + m
    return x


def dnorm(x, s=1, m=0):
    r''' The normal density function. '''
    c = (1/(np.sqrt(2*np.pi)*s))*np.exp(-0.5*((x-m)/s)**2)
    return c


def rewcorfactorlts(p, intercept, n, alpha):
    r''' Correction factor for final LTS least-squares fit.

    Args:
        1. p - [int] The rank of X, the number of parameters to fit.
        2. intercept - [int] Logical. Are you fitting an intercept?
            Set to false for array processing.
        3. n - [int] The number of data points used in processing.
        4. alpha - [float] The percentage of data points to keep in
            the LTS, i.e. h = floor(alpha*n).

    Returns:
        1. finitefactor - [float] A finite sample correction factor.

    '''

    # ALPHA = 0.500.
    coeffalpha500 = np.array([
        [-0.417574780492848, -0.175753709374146],
        [1.83958876341367, 1.8313809497999], [3, 5]])

    # ALPHA = 0.875.
    coeffalpha875 = np.array([
        [-0.267522855927958, -0.161200683014406],
        [1.17559984533974, 1.21675019853961], [3, 5]])

    # Apply eqns (6) and (7) from Pison et al. (2002).
    y1_500 = 1 + coeffalpha500[0, 0]/np.power(p, coeffalpha500[1, 0])
    y2_500 = 1 + coeffalpha500[0, 1]/np.power(p, coeffalpha500[1, 1])
    y1_875 = 1 + coeffalpha875[0, 0]/np.power(p, coeffalpha875[1, 0])
    y2_875 = 1 + coeffalpha875[0, 1]/np.power(p, coeffalpha875[1, 1])

    # Solve for new ALPHA = 0.5 coefficients for the input p.
    y1_500 = np.log(1-y1_500)
    y2_500 = np.log(1-y2_500)
    y_500 = np.array([[y1_500], [y2_500]])
    X_500 = np.array([      # noqa
        [1, np.log(1/(coeffalpha500[2, 0]*p**2))],
        [1, np.log(1/(coeffalpha500[2, 1]*p**2))]])
    c500 = np.linalg.lstsq(X_500, y_500, rcond=-1)[0]

    # Solve for new ALPHA = 0.875 coefficients for the input p.
    y1_875 = np.log(1-y1_875)
    y2_875 = np.log(1-y2_875)
    y_875 = np.array([[y1_875], [y2_875]])
    X_875 = np.array([                          # noqa
        [1, np.log(1/(coeffalpha875[2, 0]*p**2))],
        [1, np.log(1/(coeffalpha875[2, 1]*p**2))]])
    c875 = np.linalg.lstsq(X_875, y_875, rcond=-1)[0]

    # Get new correction functions for the specified n.
    fp500 = 1 - np.exp(c500[0])/np.power(n, c500[1])
    fp875 = 1 - np.exp(c875[0])/np.power(n, c875[1])

    # Linearly interpolate for the specified ALPHA.
    if (alpha >= 0.500) and (alpha <= 0.875):
        fpfinal = fp500 + ((fp875 - fp500)/0.375)*(alpha - 0.500)

    if (alpha > 0.875) and (alpha < 1):
        fpfinal = fp875 + ((1 - fp875)/0.125)*(alpha-0.875)

    finitefactor = np.asscalar(1/fpfinal)
    return finitefactor


def rewconsfactorlts(weights, n, p):
    r''' Another correction factor for the final LTS fit.

    Args:
        1. weights - [array] The standardized residuals.
        2. n - [int] The total number of data points.
        3. p - [int] The number of parameters to estimate.

    Returns:
        1. cdelta_rew - [float] A small sample correction
            factor.

    '''

    if np.sum(weights) == n:
        cdelta_rew = 1
    else:
        if p == 0:
            qdelta_rew = fltsh.qchisq(np.sum(weights)/n, 1)
            cdeltainvers_rew = fltsh.pgamma(
                qdelta_rew/2, 1.5) / (np.sum(weights) / n)
            cdelta_rew = np.sqrt(1/cdeltainvers_rew)
        else:
            a = fltsh.dnorm(1/(1/(fltsh.qnorm((sum(weights)+n)/(2*n)))))
            b = (1/fltsh.qnorm((np.sum(weights)+n)/(2*n)))
            q = 1-((2*n)/(np.sum(weights)*b))*a
            cdelta_rew = 1/np.sqrt(q)

    return cdelta_rew


def arrayfromweights(weightarray, idx):
    """ Return array element pairs from LTS weights.

    Args:
        1. weightarray - [array] An m x 0 array of the
            final LTS weights for each element pair.
        2. idx - [array] An m x 2 array of the element pairs;
            generated from the `get_cc_time` function.

    Returns:
        1. fstations - [array] A 1 x m array of element pairs.

    """

    a = np.where(weightarray == 0)[0]
    stn1, stn2 = zip(*idx)
    stn1 = np.array(stn1)
    stn2 = np.array(stn2)

    # Add one for plotting purposes; offset python 0-based indexing.
    stn1 += 1
    stn2 += 1

    # Flagged stations
    fstations = np.concatenate((stn1[a], stn2[a]))
    return fstations


def get_cc_time(data, rij, hz):
    """ Generate a time delay vector from cross correlations.

    Cross correlates data and forms the (infra/seis) co-array.

    Args:
        1. data - [array] An m x n data matrix with columns corresponding
            to different time series.
        2. rij - [array] The array coordinates.
        3. hz - [array] The sampling frequency of the data used to
            change samples to time (sec).

    Returns:
        1. tau - [array] A time delay vector for the co-array of
            the data matrix.
        2. xij - [array] The co-array of the input array.
        3. cmax - [float] The maximum of the cross-correlations.
        4. idx - [array] The co-array pairs.

    """

    m, n = np.shape(data)
    # Pre-allocate the cross-correlation matrix
    cij = np.empty((m*2-1, n))
    idx = [(i, j) for i in range(n-1) for j in range(i+1, n)]

    # Generate the co-array
    xij = rij[:, [i[0] for i in idx]] - rij[:, [j[1] for j in idx]]

    # Get time delays and number of unique inter-sensor pairs.
    N = xij.shape[1]    # noqa

    #  Pre-allocate the cross-correlation matrix
    cij = np.empty((m*2-1, N))
    for k in range(N):
        # MATLAB's xcorr w/ 'coeff' normalization: unit auto-correlations.
        cij[:, k] = (np.correlate(data[:, idx[k][0]],
                                  data[:, idx[k][1]], mode='full') / np.sqrt(
                                      sum(data[:, idx[k][0]]*data[:, idx[k][0]])
                                      * sum(data[:, idx[k][1]]*data[:, idx[k][1]])))

    # Extract cross correlation maxima and associated delays.
    cmax = cij.max(axis=0)

    # Add MATLAB-esque +1 offset here for tau.
    delay = np.argmax(cij, axis=0)+1

    # Form the time delay vector.
    tau = (m - delay)/hz

    # Reshape output matrices for next processing steps.
    xij = xij.T
    tau = np.reshape(tau, (len(tau), 1))

    return tau, xij, cmax, idx


def getrij(latlist, lonlist):
    r''' Calculate element r_{ij} from lat-lon.

    Return the projected geographic positions in X-Y (cartesian) coordinates.
    Points are calculated with the Vincenty inverse and will have a zero-mean.

    Args:
        1. latlist - A list of latitude points.
        2. lonlist - A list of longitude points.

    Returns:
        1. rij - A numpy array with the first row corresponding to
            cartesian "X" - coordinates and the second row
            corresponding to cartesian "Y" - coordinates.

    '''

    getrij.__version__ = '1.00'

    # Check that the lat-lon arrays are the same size.
    latsize = len(latlist)
    lonsize = len(lonlist)
    if latsize != lonsize:
        raise ValueError('latsize != lonsize')

    # Pre-allocate "x" and "y" arrays.
    xnew = np.zeros((latsize, ))
    ynew = np.zeros((lonsize, ))

    for jj in range(1, lonsize):
        # Obspy defaults to the WGS84 ellipsoid.
        delta, az, baz = calc_vincenty_inverse(
            latlist[0], lonlist[0], latlist[jj], lonlist[jj])
        # Convert azimuth to radians.
        az = (450 - az) % 360
        xnew[jj] = delta/1000*np.cos(az*np.pi/180)
        ynew[jj] = delta/1000*np.sin(az*np.pi/180)

    # Remove the mean.
    xnew = xnew - np.mean(xnew)
    ynew = ynew - np.mean(ynew)

    # Package as rij.
    rij = np.array([xnew.tolist(), ynew.tolist()])

    return rij


def fail_spike_test(tdelay, xij):
    r''' Returns data structures filled with nans when all tdelays are equal.

    Receiving a time delay vector where every element is equal will cause
    the current LTS algorithm to crash. In order to minimize the impact on
    existing processing pipelines, this function assists in returning
    the sample data structures as a successful LTS run, but with
    nan values replacing LTS derived parameters.

    Args:
        1. tdelay - [array] The inter-element travel times.
        2. xij - [array] The co-array coordinates.

    Returns:
        1. flagged [array] A vector of nan values.
        2. lts_estimate [dict] A collection of run paramters with
            LTS derived parameters replaced with nans.

    '''

    # Creating the "flagged" vector
    nan_vec = np.empty_like(tdelay)
    nan_vec.fill(np.nan)

    # Creating the lts_estimate, vector
    lts_estimate = {'bazimuth': np.nan, 'velocity': np.nan,
                    'coefficients': np.array([np.nan, np.nan]),
                    'flagged': nan_vec, 'fitted': nan_vec,
                    'residuals': nan_vec, 'scale': np.nan,
                    'sigma_tau': np.nan,
                    'rsquared': np.nan,
                    'X': xij, 'y': tdelay}

    LTSbaz = lts_estimate['bazimuth']
    LTSvel = lts_estimate['velocity']
    flagged = lts_estimate['flagged']

    return LTSbaz, LTSvel, flagged, lts_estimate


def least_squares_fit(Xvar, yvar, datamad, xorig, yorig):
    ''' Perform an (ordinary) least squares fit of the data.

    The simple case ALPHA == 1.0.

    Inputs:
        1. Xvar: The standardized design matrix.
        2. yvar: The standardized data array.
        3. datamad: The data median absolute deviation
            (MAD) array from standardization.
        4. xorig: The original design matrix.
            Used for post-process packaging.
        5. yorig: The original data array.
            Used for post-process packaging.

    Returns:
        1. lst_sq_estimate: The least squares fit packaged
        in a dictionary like the LTS estimate.

    '''

    # Perform the least squares fit.
    n, p = np.shape(Xvar)
    q, r = np.linalg.qr(Xvar)
    qt = q.conj().T @ yvar
    coeffs = lstsq(r, qt)[0]

    fitted = xorig @ coeffs
    residuals = yorig - fitted

    # Post-process.
    if p <= 1:
        coeffs[0] = coeffs[0] * datamad[p]/datamad[0]
    else:
        for ii in range(0, p):
            coeffs[ii] = coeffs[ii] * datamad[p] / datamad[ii]
        coeffs[p-1] = coeffs[p-1] * datamad[p-1] / datamad[p-1]

    # Calculate back-azimuth and velocity
    vel = 1/np.linalg.norm(coeffs, 2)
    # Convert baz from mathematical CCW from E
    # to geographical CW from N. baz = arctan(sx/sy)
    baz = (np.arctan2(coeffs[0], coeffs[1])*180/np.pi-360) % 360

    # Calculate the sigma_tau value (Szuberla et al. 2006).
    tdelay = np.reshape(yorig, (len(yorig), ))
    sigma_tau = np.sqrt(
        np.abs(tdelay
               @ (np.eye(n) - xorig @ np.linalg.inv(xorig.T @ xorig)
                  @ xorig.T) @ tdelay / (n - p)))

    # Packaging - Create the lst_sq_estimate packaged output.
    # Create the "flagged" vector.
    nan_vec = np.empty_like(yorig)
    nan_vec.fill(np.nan)
    lst_sq_estimate = {'bazimuth': np.asscalar(baz), 'velocity': vel,
                       'coefficients': coeffs,
                       'flagged': nan_vec, 'fitted': fitted,
                       'residuals': residuals, 'scale': np.nan,
                       'rsquared': np.nan,
                       'sigma_tau': sigma_tau,
                       'X': xorig, 'y': yorig}

    return lst_sq_estimate
