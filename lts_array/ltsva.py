import numpy as np

from lts_array.fast_lts_array import fast_lts_array
from lts_array.flts_helper_array import (get_cc_time,
                                         fail_spike_test, arrayfromweights)


def ltsva(st, rij, winlen, winover, alpha=1.0):
    r""" Process infrasound or seismic array data with least trimmed squares (LTS).

    Args:
        st: Obspy stream object. Assumes response has been removed.
        rij (2, n): Array of 'n' (infra/seis) array element coordinates
         in km for 2-dimensions [easting, northing].
        winlen (float): Window length in seconds.
        winover (float): Window overlap in the range (0.0 - 1.0).
        alpha (float): Fraction of data for LTS subsetting [0.5 - 1.0].
            Choose 1.0 for ordinary least squares (default).

    Exceptions:
        A check is performed to see if all time delays are equal (= 0).
        If so, an exception is raised and the algorithm
        returns the same data structures on exit, but with NaN values.

    Returns:
        (tuple):
            ``lts_vel`` (array): Array of least trimmed squares
            trace velocity estimates.
            ``lts_baz`` (array): Array of least trimmed squares
            back-azimuth estimates.
            ``t`` (array): Array of times at which parameter estimates
            are calculated.
            ``mdccm`` (array): Array of median cross-correlation maximas.
            ``stdict`` (dict): Dictionary of flagged element pairs.

    """

    # Pull processing parameters from the stream file.
    tvec = st[0].times('matplotlib')
    nchans = len(st)
    npts = st[0].stats.npts
    fs = st[0].stats.sampling_rate

    # check that all traces have the same length
    if len(set([len(tr) for tr in st])) != 1:
        raise ValueError('Traces in stream must have same length!')

    # Store data traces in an array for processing.
    data = np.empty((npts, nchans))
    for ii, tr in enumerate(st):
        data[:, ii] = tr.data

    # Convert window length to samples.
    winlensamp = int(winlen*fs)
    sampinc = int((1-winover)*winlensamp)
    its = np.arange(0, npts, sampinc)
    nits = len(its) - 1

    # Pre-allocate data arrays.
    mdccm = np.full(nits, np.nan)
    t = np.full(nits, np.nan)
    lts_vel = np.full(nits, np.nan)
    lts_baz = np.full(nits, np.nan)
    sigma_tau = np.full(nits, np.nan)

    # State if least trimmed squares or ordinary least squares will be used.
    if alpha == 1.0:
        print('ALPHA is 1.00. Performing an ordinary',
              'least squares fit, NOT least trimmed squares.')
        print('Calculating sigma_tau.')

    # Station dictionary for dropped LTS elements.
    stdict = {}

    counter = 0

    # Loop through the time series.
    print('Running ltsva for %d windows' % nits)
    for jj in range(nits):

        # Get time from middle of window, except for the end.
        ptr = int(its[jj]), int(its[jj] + winlensamp)
        try:
            t[jj] = tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[jj] = np.nanmax(t, axis=0)

        # Cross correlate the wave forms. Get the differential times.
        tdelay, xij, ccmax, idx = get_cc_time(data[ptr[0]:ptr[1], :], rij, fs)

        # Check to see if time delays are all equal (= 0).
        # The fast_lts_array function will crash if all time
        # delays (tdelay) are equal. In our experience,
        # this can occur if electronic spikes are present
        # in the data.
        # Return data structure filled with NaNs if true.
        dataspike = np.all(tdelay == 0)
        if dataspike:
            print("Tdelays are equal. LTS algorithm not run. \
                                    Returning NaNs for LTS output terms.")
            lts_baz[jj], lts_vel[jj], flagged, lts_estimate = fail_spike_test(
                tdelay, xij)
            sigma_tau[jj] = lts_estimate['sigma_tau']
            mdccm[jj] = np.nan

            # keep track of progress
            counter += 1
            continue

        else:
            # Apply the FAST-LTS algorithm.
            lts_estimate = fast_lts_array(xij, tdelay, alpha)

        lts_baz[jj] = lts_estimate['bazimuth']
        lts_vel[jj] = lts_estimate['velocity']
        sigma_tau[jj] = lts_estimate['sigma_tau']
        mdccm[jj] = np.median(ccmax)

        # Map dropped data points back to elements.
        stns = arrayfromweights(lts_estimate['flagged'], idx)

        # Stash the number of elements for plotting.
        if len(stns) > 0:
            tval = str(t[jj])
            stdict[tval] = stns
        if jj == (nits-1) and alpha != 1.0:
            stdict['size'] = nchans

        # Print progress
        counter += 1
        print('{:.1f}%'.format((counter / nits) * 100), end='\r')

    print('\nDone\n')

    return lts_vel, lts_baz, t, mdccm, stdict, sigma_tau
