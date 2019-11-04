from matplotlib import dates
import numpy as np
import sys

from fast_lts_array import fast_lts_array
from flts_helper_array import get_cc_time, fail_spike_test, arrayfromweights


def ltsva(st, rij, WINLEN, WINOVER, ALPHA):
    r""" Process infrasound or seismic array data with least trimmed squares (LTS)

    Args:
        1. filtered obspy stream. Assumes response has been removed.
        2. rij - [array] - (2, n) 'n' array element coordinates in km
                for 2-dimenensions [easting, northing]
        3. WINLEN - [float] - window length in seconds
        4. WINOVER - [float] - window overlap [<1.0]
        5. alpha - [float] - fraction of data for LTS subsetting [0.5, 1.0]

    Exceptions:
        1. Exception - A check is performed to see if all time delays
            are equal. If so, an exception is raised and the algorithm
            on exit returns the same data structures, but with NaN values.

    Returns:
        stdict: dictionary of flagged element pairs
        t: array processing time vector
        mdccm: median cross-correlation maxima
        LTSvel: least-trimmed squares trace velocity
        LTSbaz: least-trimmed squares back-azimuth

            #should we keep these for a description perhaps???
            3. flagged - [array] The binary (0 or 1) weights assigned
            to station pairs in the final weighted least squares fit.
            Stations with a final weight of "0" are flagged as outlying
            4. ccmax - [array] The cross correlation maxima used to
              determine inter-element travel times.
            5. idx - [array] Station pairs.
            6. lts_estimate [dictionary] A dictionary with the following keys:
                a. bazimuth - [float] fltsbaz; the back-azimuth in
                    degrees from north as determined by the
                    least trimmed squares fit.
                b. velocity - [float] fltsvel; the velocity determined
                    by the least trimmed squares fit.
                c. coefficients - [array] The x and y components of
                    the slowness vector [sx, sy].
                d. flagged - [array] The binary (0 or 1) weights assigned
                    to station pairs in the final weighted least squares fit.
                    Stations with a final weight of "0" are flagged as outlying
                    by the algorithm.
                e. fitted - [array] The value of the best-fit plane at
                    the co-array coordinates.
                f. residuals - [array] - The residuals between the
                    "fitted" values and the "y" values.
                g. scale - [float] The scale value used to
                    determine the LTS weights.
                h. rsquared - [float] The R**2 value of the regression fit.
                j. X - [array] The input co-array coordinate array.
                k. y - [array] The input inter-element travel times.

    """

    # Parameters from the stream file
    tvec = dates.date2num(st[0].stats.starttime.datetime)+st[0].times()/86400
    nchans = len(st)
    npts = st[0].stats.npts
    fs = st[0].stats.sampling_rate

    data = np.empty((npts, nchans))
    for ii, tr in enumerate(st):
        data[:, ii] = tr.data

    winlensamp = int(WINLEN*fs)
    sampinc = int((1-WINOVER)*winlensamp)
    its = np.arange(0, npts, sampinc)
    nits = len(its)-1

    # Pre-allocating Data Arrays
    mdccm = np.full(nits, np.nan)
    t = np.full(nits, np.nan)
    LTSvel = np.full(nits, np.nan)
    LTSbaz = np.full(nits, np.nan)

    # Station Dictionary for Dropped LTS Elements
    stdict = {}

    print('Running ltsva for %d windows' % nits)
    for jj in range(nits):

        # Get time from middle of window, except for the end
        ptr = int(its[jj]), int(its[jj] + winlensamp)
        try:
            t[jj] = tvec[ptr[0]+int(winlensamp/2)]
        except:
            t[jj] = np.nanmax(t, axis=0)

        tdelay, xij, ccmax, idx = get_cc_time(data[ptr[0]:ptr[1], :], rij, fs)

        # Check to see if time delays are all equal. The fastlts
        #  function will crash if all tdelays are equal.
        # Return data structures filled with nans if true.
        dataspike = np.all(tdelay == 0)
        if dataspike:
            raise Exception("Tdelays are equal. LTS algorithm not run. \
                                    Returning NaNs for LTS output terms.")
            fltsbaz, fltsvel, flagged, lts_estimate = fail_spike_test(
                tdelay, xij)
            return fltsbaz, fltsvel, flagged, ccmax, idx, lts_estimate

        # Apply the FAST-LTS algorithm and return results
        if ALPHA == 1.0:
            print('ALPHA is 1.00. Performing an ordinary',
                  ' least squares fit, NOT least trimmed squares.')
        lts_estimate = fast_lts_array(xij, tdelay, ALPHA)

        LTSbaz[jj] = lts_estimate['bazimuth']
        LTSvel[jj] = lts_estimate['velocity']

        mdccm[jj] = np.median(ccmax)
        stns = arrayfromweights(lts_estimate['flagged'], idx)

        # Stash some metadata for plotting
        if len(stns) > 0:
            tval = str(t[jj])
            stdict[tval] = stns
        if jj == (nits-1):
            stdict['size'] = nchans

        tmp = int(jj/nits*100)
        sys.stdout.write("\r%d%% \n" % tmp)
        sys.stdout.flush()
    print('Done\n')

    return stdict, t, mdccm, LTSvel, LTSbaz
