import numpy as np
from fast_lts_array import fastlts
from flts_helper_array import get_cc_time
from flts_helper_array import fail_spike_test


def ltsva(data, rij, fs, alpha):
    r""" Process infrasound and seismic array data with least trimmed squares (LTS)

    @author: Jordan W Bishop

    Args:
        1. data - [array] - (m, n) Array of time series with for
            cross correlations with 'm' samples and
            'n' waveforms as columns
        2. fs - [float] - sampling rate
        3. rij - [array] - (2, n) 'n' array element coordinates in km
        in 2-dimenensions [easting, northing]
        4. alpha - [float] - fraction of data for LTS subsetting [0.5, 1.0]

    Exceptions:
        1. Exception - A check is performed to see if all time delays
            are equal. If so, an exception is raised and the algorithm
            on exit returns the same data structures, but with NaN values.

    Returns:
            1. fltsbaz - [float] The back-azimuth in degrees from north as
              determined by the least trimmed squares fit.
            2. fltsvel - [float] The velocity determined by
                the least trimmed squares fit.
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

    # Cross-correlate station pairs to determine inter-element time delays
    tdelay, xij, ccmax, idx = get_cc_time(data, rij, fs)

    # Check to see if time delays are all equal. The fastlts
    #  function will crash if all tdelays are equal.
    # Return data structures filled with nans if true.
    dataspike = np.all(tdelay == 0)
    if dataspike:
        raise Exception("Tdelays are equal. LTS algorithm not run. \
                                Returning NaNs for LTS output terms.")
        fltsbaz, fltsvel, flagged, lts_estimate = fail_spike_test(tdelay, xij)
        return fltsbaz, fltsvel, flagged, ccmax, idx, lts_estimate

    # Apply the FAST-LTS algorithm and return results
    lts_estimate = fastlts(xij, tdelay, alpha)

    fltsbaz = lts_estimate['bazimuth']
    fltsvel = lts_estimate['velocity']
    flagged = lts_estimate['flagged']

    return fltsbaz, fltsvel, flagged, ccmax, idx, lts_estimate
