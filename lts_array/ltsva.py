import numpy as np
import sys

from fast_lts_array import fast_lts_array
from flts_helper_array import get_cc_time, fail_spike_test, arrayfromweights


def ltsva(st, rij, WINLEN, WINOVER, ALPHA):
    r""" Process infrasound or seismic array data
                                            with least trimmed squares (LTS).

    Args:
        1. st - Obspy stream object. Assumes response has been removed.
        2. rij - (2, n) Array of 'n' (infra/seis) array element coordinates
         in km for 2-dimensions [easting, northing].
        3. WINLEN - Window length [float] in seconds.
        4. WINOVER - Window overlap [float] in the range [0.0 - 1.0].
        5. ALPHA - Fraction of data [float] for LTS subsetting [0.5 - 1.0].

    Exceptions:
        1. Exception - A check is performed to see if all time delays
            are equal (= 0). If so, an exception is raised and the algorithm
            returns the same data structures on exit, but with NaN values.

    Returns:
        1. stdict: Dictionary of flagged element pairs.
        2. t: Array of times at which parameter estimates are calculated.
        3. mdccm: Array of median cross-correlation maximas.
        4. lts_vel: Array of least trimmed squares trace velocity estimates.
        5. lts_baz: Array of least trimmed squares back-azimuth estimates.

    """

    # Pull processing parameters from the stream file.
    tvec = st[0].times('matplotlib')
    nchans = len(st)
    npts = st[0].stats.npts
    fs = st[0].stats.sampling_rate

    # Store data traces in an array for processing.
    data = np.empty((npts, nchans))
    for ii, tr in enumerate(st):
        data[:, ii] = tr.data

    # Convert window length to samples.
    winlensamp = int(WINLEN*fs)
    sampinc = int((1-WINOVER)*winlensamp)
    its = np.arange(0, npts, sampinc)
    nits = len(its) - 1

    # Pre-allocate data arrays.
    mdccm = np.full(nits, np.nan)
    t = np.full(nits, np.nan)
    lts_vel = np.full(nits, np.nan)
    lts_baz = np.full(nits, np.nan)
    sigma_tau = np.full(nits, np.nan)

    # State if least trimmed squares or ordinary least squares will be used.
    if ALPHA == 1.0:
        print('ALPHA is 1.00. Performing an ordinary',
              ' least squares fit, NOT least trimmed squares.')
        print('Calculating sigma_tau.')

    # Station dictionary for dropped LTS elements.
    stdict = {}

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

        """ Check to see if time delays are all equal (= 0).
        The fast_lts_array function will crash if all time
        delays (tdelay) are equal. In our experience,
        this can occur if electronic spikes are present
        in the data.
        Return data structure filled with NaNs if true. """
        dataspike = np.all(tdelay == 0)
        if dataspike:
            raise Exception("Tdelays are equal. LTS algorithm not run. \
                                    Returning NaNs for LTS output terms.")
            lts_baz, lts_vel, flagged, lts_estimate = fail_spike_test(
                tdelay, xij)
            return lts_baz, lts_vel, flagged, ccmax, idx, lts_estimate

        # Apply the FAST-LTS algorithm.
        lts_estimate = fast_lts_array(xij, tdelay, ALPHA)

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
        if jj == (nits-1):
            stdict['size'] = nchans

        tmp = int(jj/nits*100)
        sys.stdout.write("\r%d%%" % tmp)
        sys.stdout.flush()
    print('Done\n')

    return stdict, t, mdccm, lts_vel, lts_baz, sigma_tau
