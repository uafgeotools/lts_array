import numpy as np
from scipy.linalg import lstsq
from scipy.special import erfinv
from numba import jit

# Layout:
# 1) jit-decorated functions
# 2) functions
# 3) class definitions

########################
# jit-decorated functions
########################


@jit(nopython=True)
def random_set(tot, npar, seed):
    """ Generate a random data subset for LTS. """
    randlist = []
    for ii in range(0, npar):
        seed = np.floor(seed * 5761) + 999
        quot = np.floor(seed / 65536)
        seed = np.floor(seed) - np.floor(quot * 65536)
        random = float(seed / 65536)
        num = np.floor(random * tot)
        if ii > 0:
            while num in randlist:
                seed = np.floor(seed * 5761) + 999
                quot = np.floor(seed / 65536)
                seed = np.floor(seed) - np.floor(quot * 65536)
                random = float(seed / 65536)
                num = np.floor(random * tot)
        randlist.append(num)
    randset = np.array(randlist, dtype=np.int64)

    return randset, np.int64(seed)


@jit(nopython=True)
def check_array(candidate_size, best_coeff, best_obj, z, obj):
    """ Keep best coefficients for final C-step iteration.
    Don't keep duplicates.
    """
    insert = True
    for kk in range(0, candidate_size):
        if (best_obj[kk] == obj) and ((best_coeff[:, kk] == z).all()):
            insert = False
    if insert:
        bobj = np.concatenate((best_obj, np.array([obj])))
        bcoeff = np.concatenate((best_coeff, z), axis=1)
        idx = np.argsort(bobj)[0:candidate_size]
        bobj = bobj[idx]
        bcoeff = bcoeff[:, idx]

    return bcoeff, bobj


@jit(nopython=True)
def fast_LTS(nits, tau, time_delay_mad, xij_standardized, xij_mad, dimension_number, candidate_size, n_samples, co_array_num, slowness_coeffs, csteps, h, csteps2, random_set, _insertion):
    """ Run the FAST_LTS algorithm to determine an initial optimal slowness vector.
    """
    for jj in range(nits):

        # Check for data spike.
        if (time_delay_mad[jj] == 0) or (np.count_nonzero(tau[:, jj, :]) < (co_array_num - 2)):
            # We have a data spike, so do not process.
            continue

        # Standardize the y-values
        y_var = tau[:, jj, :] / time_delay_mad[jj]
        X_var = xij_standardized

        objective_array = np.full((candidate_size, ), np.inf)
        coeff_array = np.full((dimension_number, candidate_size), np.nan) # noqa
        # Initial seed for random search
        seed = 0
        # Initial best objective function value
        best_objective = np.inf
        # Initial search through the subsets
        for ii in range(0, n_samples):
            prev_obj = 0
            # Initial random solution
            index, seed = random_set(co_array_num, dimension_number, seed) # noqa
            q, r = np.linalg.qr(X_var[index, :])
            qt = q.conj().T @ y_var[index]
            z = np.linalg.lstsq(r, qt)[0]
            residuals = y_var - X_var @ z
            # Perform C-steps
            for kk in range(0, csteps):
                sortind = np.argsort(np.abs(residuals).flatten()) # noqa
                obs_in_set = sortind.flatten()[0:h]
                q, r = np.linalg.qr(X_var[obs_in_set, :])
                qt = q.conj().T @ y_var[obs_in_set]
                z = np.linalg.lstsq(r, qt)[0]
                residuals = y_var - X_var @ z
                # Sort the residuals in magnitude from low to high
                sor = np.sort(np.abs(residuals).flatten()) # noqa
                # Sum the first "h" squared residuals
                obj = np.sum(sor[0:h]**2)
                # Stop if the C-steps have converged
                if (kk >= 1) and (obj == prev_obj):
                    break
                prev_obj = obj
            # Save these initial estimates for future C-steps in next round # noqa
            if obj < np.max(objective_array):
                # Save the best objective function values.
                coeff_array, objective_array = check_array(
                    candidate_size, coeff_array, objective_array, z, obj) # noqa

        # Final condensation of promising data points
        for ii in range(0, candidate_size):
            prev_obj = 0
            if np.isfinite(objective_array[ii]):
                z = coeff_array[:, ii].copy()
                z = z.reshape((len(z), 1))
            else:
                index, seed = random_set(co_array_num, dimension_number, seed) # noqa
                q, r = np.linalg.qr(X_var[index, :])
                qt = q.conj().T @ y_var[index]
                z = np.linalg.lstsq(r, qt)[0]

            if np.isfinite(z[0]):
                residuals = y_var - X_var @ z
                # Perform C-steps
                for kk in range(0, csteps2):
                    sort_ind = np.argsort(np.abs(residuals).flatten()) # noqa
                    obs_in_set = sort_ind.flatten()[0:h]
                    q, r = np.linalg.qr(X_var[obs_in_set, :])
                    qt = q.conj().T @ y_var[obs_in_set]
                    z = np.linalg.lstsq(r, qt)[0]
                    residuals = y_var - X_var @ z
                    # Sort the residuals in magnitude from low to high
                    sor = np.sort(np.abs(residuals).flatten()) # noqa
                    # Sum the first "h" squared residuals
                    obj = np.sum(sor[0:h]**2)
                    # Stop if the C-steps have converged
                    if (kk >= 1) and (obj == prev_obj):
                        break
                    prev_obj = obj
                if obj < best_objective:
                    best_objective = obj
                    coeffs = z.copy()

        # Correct coefficients due to standardization
        for ii in range(0, dimension_number):
            coeffs[ii] *= time_delay_mad[jj] / xij_mad[ii]

        slowness_coeffs[:, jj] = coeffs.flatten()

    return slowness_coeffs


###########
# functions
###########
def raw_corfactor_lts(p, n, ALPHA):
    r""" Calculates the correction factor (from Pison et al. 2002)
        to make the LTS solution unbiased for small n.

    Args:
        p (int): The rank of X, the number of parameters to fit.
        n (int): The number of data points used in processing.
        ALPHA (float): The percentage of data points to keep in
            the LTS, e.g. h = floor(ALPHA*n).

    Returns:
        (float):
        ``finitefactor``: A correction factor to make the LTS
        solution approximately unbiased for small (i.e. finite n).

    """

    # ALPHA = 0.875.
    coeffalpha875 = np.array([
        [-0.251778730491252, -0.146660023184295],
        [0.883966931611758, 0.86292940340761], [3, 5]])
    # ALPHA = 0.500.
    coeffalpha500 = np.array([
        [-0.487338281979106, -0.340762058011],
        [0.405511279418594, 0.37972360544988], [3, 5]])

    # Apply eqns (6) and (7) from Pison et al. (2002)
    y1_500 = 1 + coeffalpha500[0, 0] / np.power(p, coeffalpha500[1, 0])
    y2_500 = 1 + coeffalpha500[0, 1] / np.power(p, coeffalpha500[1, 1])
    y1_875 = 1 + coeffalpha875[0, 0] / np.power(p, coeffalpha875[1, 0])
    y2_875 = 1 + coeffalpha875[0, 1] / np.power(p, coeffalpha875[1, 1])

    # Solve for new ALPHA = 0.5 coefficients for the input p.
    y1_500 = np.log(1 - y1_500)
    y2_500 = np.log(1 - y2_500)
    y_500 = np.array([[y1_500], [y2_500]])
    X_500 = np.array([      # noqa
        [1, np.log(1/(coeffalpha500[2, 0]*p**2))],
        [1, np.log(1/(coeffalpha500[2, 1]*p**2))]])
    c500 = np.linalg.lstsq(X_500, y_500, rcond=-1)[0]

    # Solve for new ALPHA = 0.875 coefficients for the input p.
    y1_875 = np.log(1 - y1_875)
    y2_875 = np.log(1 - y2_875)
    y_875 = np.array([[y1_875], [y2_875]])
    X_875 = np.array([      # noqa
        [1, np.log(1 / (coeffalpha875[2, 0] * p**2))],
        [1, np.log(1 / (coeffalpha875[2, 1] * p**2))]])
    c875 = np.linalg.lstsq(X_875, y_875, rcond=-1)[0]

    # Get new correction factors for the specified n.
    fp500 = 1 - np.exp(c500[0]) / np.power(n, c500[1])
    fp875 = 1 - np.exp(c875[0]) / np.power(n, c875[1])

    # Linearly interpolate for the specified ALPHA.
    if (ALPHA >= 0.500) and (ALPHA <= 0.875):
        fpfinal = fp500 + ((fp875 - fp500) / 0.375)*(ALPHA - 0.500)

    if (ALPHA > 0.875) and (ALPHA < 1):
        fpfinal = fp875 + ((1 - fp875) / 0.125)*(ALPHA - 0.875)

    finitefactor = np.ndarray.item(1 / fpfinal)
    return finitefactor


def raw_consfactor_lts(h, n):
    r""" Calculate the constant used to make the
     LTS scale estimators consistent for
     a normal distribution.

    Args:
        h (int): The number of points to fit.
        n (int): The total number of data points.

    Returns:
        (float):
        ``dhn``: The correction factor d_h,n.

    """
    # Calculate the initial factor c_h,n.
    x = (h + n) / (2 * n)
    phinv = np.sqrt(2) * erfinv(2 * x - 1)
    chn = 1 / phinv

    # Calculate d_h,n.
    phi = (1 / np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * phinv**2)
    d = np.sqrt(1 - (2 * n / (h * chn)) * phi)
    dhn = 1 / d

    return dhn


def _qnorm(p, s=1, m=0):
    r""" The normal inverse distribution function. """
    x = erfinv(2 * p - 1) * np.sqrt(2) * s + m
    return x


def _dnorm(x, s=1, m=0):
    r""" The normal density function. """
    c = (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s)**2)
    return c


def rew_corfactor_lts(p, n, ALPHA):
    r""" Correction factor for final LTS least-squares fit.

    Args:
        p (int): The rank of X, the number of parameters to fit.
        intercept (int): Logical. Are you fitting an intercept?
            Set to false for array processing.
        n (int): The number of data points used in processing.
        ALPHA (float): The percentage of data points to keep in
            the LTS, e.g. h = floor(ALPHA*n).

    Returns:
        (float):
        ``finitefactor``: A finite sample correction factor.

    """

    # ALPHA = 0.500.
    coeffalpha500 = np.array([
        [-0.417574780492848, -0.175753709374146],
        [1.83958876341367, 1.8313809497999], [3, 5]])

    # ALPHA = 0.875.
    coeffalpha875 = np.array([
        [-0.267522855927958, -0.161200683014406],
        [1.17559984533974, 1.21675019853961], [3, 5]])

    # Apply eqns (6) and (7) from Pison et al. (2002).
    y1_500 = 1 + coeffalpha500[0, 0] / np.power(p, coeffalpha500[1, 0])
    y2_500 = 1 + coeffalpha500[0, 1] / np.power(p, coeffalpha500[1, 1])
    y1_875 = 1 + coeffalpha875[0, 0] / np.power(p, coeffalpha875[1, 0])
    y2_875 = 1 + coeffalpha875[0, 1] / np.power(p, coeffalpha875[1, 1])

    # Solve for new ALPHA = 0.5 coefficients for the input p.
    y1_500 = np.log(1 - y1_500)
    y2_500 = np.log(1 - y2_500)
    y_500 = np.array([[y1_500], [y2_500]])
    X_500 = np.array([      # noqa
        [1, np.log(1 / (coeffalpha500[2, 0] * p**2))],
        [1, np.log(1 / (coeffalpha500[2, 1] * p**2))]])
    c500 = np.linalg.lstsq(X_500, y_500, rcond=-1)[0]

    # Solve for new ALPHA = 0.875 coefficients for the input p.
    y1_875 = np.log(1 - y1_875)
    y2_875 = np.log(1 - y2_875)
    y_875 = np.array([[y1_875], [y2_875]])
    X_875 = np.array([                          # noqa
        [1, np.log(1 / (coeffalpha875[2, 0] * p**2))],
        [1, np.log(1 / (coeffalpha875[2, 1] * p**2))]])
    c875 = np.linalg.lstsq(X_875, y_875, rcond=-1)[0]

    # Get new correction functions for the specified n.
    fp500 = 1 - np.exp(c500[0]) / np.power(n, c500[1])
    fp875 = 1 - np.exp(c875[0]) / np.power(n, c875[1])

    # Linearly interpolate for the specified ALPHA.
    if (ALPHA >= 0.500) and (ALPHA <= 0.875):
        fpfinal = fp500 + ((fp875 - fp500) / 0.375)*(ALPHA - 0.500)

    if (ALPHA > 0.875) and (ALPHA < 1):
        fpfinal = fp875 + ((1 - fp875) / 0.125)*(ALPHA - 0.875)

    finitefactor = np.ndarray.item(1 / fpfinal)
    return finitefactor


def rew_consfactor_lts(weights, p, n):
    r""" Another correction factor for the final LTS fit.

    Args:
        weights (array): The standardized residuals.
        n (int): The total number of data points.
        p (int): The number of parameters to estimate.

    Returns:
        (float):
        ``cdelta_rew``: A small sample correction factor.

    """
    a = _dnorm(1 / (1 / (_qnorm((sum(weights) + n) / (2 * n)))))
    b = (1 / _qnorm((np.sum(weights) + n) / (2 * n)))
    q = 1 - ((2 * n) / (np.sum(weights) * b)) * a
    cdelta_rew = 1/np.sqrt(q)

    return cdelta_rew


def cubicEqn(a, b, c):
    r"""
    Roots of cubic equation in the form :math:`x^3 + ax^2 + bx + c = 0`.

    Args:
        a (int or float): Scalar coefficient of cubic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above

    Returns:
        list: Roots of cubic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Relatively stable solutions, with some tweaks by Dr. Z,
        per algorithm of Numerical Recipes 2nd ed., :math:`\S` 5.6. Even
        :func:`numpy.roots` can have some (minor) issues; e.g.,
        :math:`x^3 - 5x^2 + 8x - 4 = 0`.
    """

    Q = a*a/9 - b/3
    R = (3*c - a*b)/6 + a*a*a/27
    Q3 = Q*Q*Q
    R2 = R*R
    ao3 = a/3

    # Q & R are real
    if np.isreal([a, b, c]).all():
        # 3 real roots
        if R2 < Q3:
            sqQ = -2 * np.sqrt(Q)
            theta = np.arccos(R / np.sqrt(Q3))
            # This solution first published in 1615 by Viète!
            x = [sqQ * np.cos(theta / 3) - ao3,
                 sqQ * np.cos((theta + 2 * np.pi) / 3) - ao3,
                 sqQ * np.cos((theta - 2 * np.pi) / 3) - ao3]
        # Q & R real, but 1 real, 2 complex roots
        else:
            # this is req'd since np.sign(0) = 0
            if R != 0:
                A = -np.sign(R) * (np.abs(R) + np.sqrt(R2 - Q3)) ** (1 / 3)
            else:
                A = -np.sqrt(-Q3) ** (1 / 3)
            if A == 0:
                B = 0
            else:
                B = Q/A
            # one real root & two conjugate complex ones
            x = [
                (A+B) - ao3,
                -.5 * (A+B) + 1j * np.sqrt(3) / 2 * (A - B) - ao3,
                -.5 * (A+B) - 1j * np.sqrt(3) / 2 * (A - B) - ao3]
    # Q & R complex, so also 1 real, 2 complex roots
    else:
        sqR2mQ3 = np.sqrt(R2 - Q3)
        if np.real(np.conj(R) * sqR2mQ3) >= 0:
            A = -(R+sqR2mQ3)**(1/3)
        else:
            A = -(R-sqR2mQ3)**(1/3)
        if A == 0:
            B = 0
        else:
            B = Q/A
        # one real root & two conjugate complex ones
        x = [
            (A+B) - ao3,
            -.5 * (A+B) + 1j * np.sqrt(3) / 2 * (A - B) - ao3,
            -.5 * (A+B) - 1j * np.sqrt(3) / 2 * (A - B) - ao3
                ]
    # parse real and/or int roots for tidy output
    for k in range(0, 3):
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def quadraticEqn(a, b, c):
    r"""
    Roots of quadratic equation in the form :math:`ax^2 + bx + c = 0`.

    Args:
        a (int or float): Scalar coefficient of quadratic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above

    Returns:
        list: Roots of quadratic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Stable solutions, even for :math:`b^2 >> ac` or complex coefficients,
        per algorithm of Numerical Recipes 2nd ed., :math:`\S` 5.6.
    """

    # real coefficient branch
    if np.isreal([a, b, c]).all():
        # note np.sqrt(-1) = nan, so force complex argument
        if b:
            # std. sub-branch
            q = -0.5*(b + np.sign(b) * np.sqrt(complex(b * b - 4 * a * c)))
        else:
            # b = 0 sub-branch
            q = -np.sqrt(complex(-a * c))
    # complex coefficient branch
    else:
        if np.real(np.conj(b) * np.sqrt(b * b - 4 * a * c)) >= 0:
            q = -0.5*(b + np.sqrt(b * b - 4 * a * c))
        else:
            q = -0.5*(b - np.sqrt(b * b - 4 * a * c))
    # stable root solution
    x = [q/a, c/q]
    # parse real and/or int roots for tidy output
    for k in 0, 1:
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])
    return x


def quarticEqn(a, b, c, d):
    r"""
    Roots of quartic equation in the form :math:`x^4 + ax^3 + bx^2 +
    cx + d = 0`.

    Args:
        a (int or float): Scalar coefficient of quartic equation, can be
            complex
        b (int or float): Same as above
        c (int or float): Same as above
        d (int or float): Same as above

    Returns:
        list: Roots of quartic equation in standard form

    See Also:
        :func:`numpy.roots` — Generic polynomial root finder

    Notes:
        Stable solutions per algorithm of CRC Std. Mathematical Tables, 29th
        ed.
    """

    # find *any* root of resolvent cubic
    a2 = a*a
    y = cubicEqn(-b, a*c - 4*d, (4*b - a2)*d - c*c)
    y = y[0]
    # find R
    R = np.sqrt(a2 / 4 - (1 + 0j) * b + y)  # force complex in sqrt
    foo = 3*a2/4 - R*R - 2*b
    if R != 0:
        # R is already complex.
        D = np.sqrt(foo + (a * b - 2 * c - a2 * a / 4) / R)
        E = np.sqrt(foo - (a * b - 2 * c - a2 * a / 4) / R)  # ...
    else:
        sqrtTerm = 2 * np.sqrt(y * y - (4 + 0j) * d)  # force complex in sqrt
        D = np.sqrt(foo + sqrtTerm)
        E = np.sqrt(foo - sqrtTerm)
    x = [-a/4 + R/2 + D/2,
         -a/4 + R/2 - D/2,
         -a/4 - R/2 + E/2,
         -a/4 - R/2 - E/2]
    # parse real and/or int roots for tidy output
    for k in range(0, 4):
        if np.real(x[k]) == x[k]:
            x[k] = float(np.real(x[k]))
            if int(x[k]) == x[k]:
                x[k] = int(x[k])

    return x


def rthEllipse(a, b, x0, y0):
    r"""
    Calculate angles subtending, and extremal distances to, a
    coordinate-aligned ellipse from the origin.

    Args:
        a (float): Semi-major axis of ellipse
        b (float): Semi-minor axis of ellipse
        x0 (float): Horizontal center of ellipse
        y0 (float): Vertical center of ellipse

    Returns:
        tuple: Tuple containing:

        - **eExtrm** – Extremal parameters in ``(4, )`` array as

          .. code-block:: none

            [min distance, max distance, min angle (degrees), max angle (degrees)]

        - **eVec** – Coordinates of extremal points on ellipse in ``(4, 2)``
          array as

          .. code-block:: none

            [[x min dist., y min dist.],
             [x max dist., y max dist.],
             [x max angle tangency, y max angle tangency],
             [x min angle tangency, y min angle tangency]]
    """

    # set constants
    A = 2/a**2
    B = 2*x0/a**2
    C = 2/b**2
    D = 2*y0/b**2
    E = (B*x0+D*y0)/2-1
    F = C-A
    G = A/2
    H = C/2
    eExtrm = np.zeros((4,))
    eVec = np.zeros((4, 2))
    eps = np.finfo(np.float64).eps

    # some tolerances for numerical errors
    circTol = 1e8  # is it circular to better than circTol*eps?
    zeroTol = 1e4  # is center along a coord. axis to better than zeroTol*eps?
    magTol = 1e-5  # is a sol'n within ellipse*(1+magTol) (magnification)

    # pursue circular or elliptical solutions
    if np.abs(F) <= circTol * eps:
        # circle
        cent = np.sqrt(x0 ** 2 + y0 ** 2)
        eExtrm[0:2] = cent + np.array([-a, a])
        eVec[0:2, :] = np.array([
            [x0-a*x0/cent, y0-a*y0/cent],
            [x0+a*x0/cent, y0+a*y0/cent]])
    else:
        # ellipse
        # check for trivial distance sol'n
        if np.abs(y0) < zeroTol * eps:
            eExtrm[0:2] = x0 + np.array([-a, a])
            eVec[0:2, :] = np.vstack((eExtrm[0:2], [0, 0])).T
        elif np.abs(x0) < zeroTol * eps:
            eExtrm[0:2] = y0 + np.array([-b, b])
            eVec[0:2, :] = np.vstack(([0, 0], eExtrm[0:2])).T
        else:
            # use dual solutions of quartics to find best, real-valued results
            # solve quartic for y
            fy = F**2*H
            y = quarticEqn(-D*F*(2*H+F)/fy,
                           (B**2*(G+F)+E*F**2+D**2*(H+2*F))/fy,
                           -D*(B**2+2*E*F+D**2)/fy, (D**2*E)/fy)
            y = np.array([y[i] for i in list(np.where(y == np.real(y))[0])])
            xy = B*y / (D-F*y)
            # solve quartic for x
            fx = F**2*G
            x = quarticEqn(B*F*(2*G-F)/fx, (B**2*(G-2*F)+E*F**2+D**2*(H-F))/fx,
                           B*(2*E*F-B**2-D**2)/fx, (B**2*E)/fx)
            x = np.array([x[i] for i in list(np.where(x == np.real(x))[0])])
            yx = D*x / (F*x+B)
            # combine both approaches
            distE = np.hstack(
                (np.sqrt(x ** 2 + yx ** 2), np.sqrt(xy ** 2 + y ** 2)))
            # trap real, but bogus sol's (esp. near Th = 180)
            distEidx = np.where(
                (distE <= np.sqrt(x0 ** 2 + y0 ** 2)
                 + np.max([a, b]) * (1 + magTol))
                & (distE >= np.sqrt(x0 ** 2 + y0 ** 2)
                   - np.max([a, b]) * (1 + magTol)))
            coords = np.hstack(((x, yx), (xy, y))).T
            coords = coords[distEidx, :][0]
            distE = distE[distEidx]
            eExtrm[0:2] = [distE.min(), distE.max()]
            eVec[0:2, :] = np.vstack(
                (coords[np.where(distE == distE.min()), :][0][0],
                 coords[np.where(distE == distE.max()), :][0][0]))
    # angles subtended
    if x0 < 0:
        x0 = -x0
        y = -np.array(quadraticEqn(D ** 2 + B ** 2 * H / G, 4 * D * E,
                                   4 * E ** 2 - B ** 2 * E / G))
        x = -np.sqrt(E / G - H / G * y ** 2)
    else:
        y = -np.array(quadraticEqn(D ** 2 + B ** 2 * H / G, 4 * D * E,
                                   4 * E ** 2 - B ** 2 * E / G))
        x = np.sqrt(E / G - H / G * y ** 2)
    eVec[2:, :] = np.vstack((np.real(x), np.real(y))).T
    # various quadrant fixes
    if x0 == 0 or np.abs(x0) - a < 0:
        eVec[2, 0] = -eVec[2, 0]
    eExtrm[2:] = np.sort(np.arctan2(eVec[2:, 1], eVec[2:, 0]) / np.pi * 180)

    return eExtrm, eVec


def post_process(dimension_number, co_array_num, alpha, h, nits, tau, xij, coeffs, lts_vel, lts_baz, element_weights, sigma_tau, p, conf_int_vel, conf_int_baz):

    # Initial fit - correction factor to make LTS approximately unbiased
    raw_factor = raw_corfactor_lts(dimension_number, co_array_num, alpha)
    # Initial fit - correction factor to make LTS approximately normally distributed # noqa
    raw_factor *= raw_consfactor_lts(h, co_array_num)
    # Final fit - correction factor to make LTS approximately unbiased
    rew_factor1 = rew_corfactor_lts(dimension_number, co_array_num, alpha)
    # Value of the normal inverse distribution function
    # at 0.9875 (98.75%)
    quantile = 2.2414027276049473
    # Co-array
    X_var = xij
    # Chi^2; default is 90% confidence (p = 0.90)
    # Special closed form for 2 degrees of freedom
    chi2 = -2 * np.log(1 - p)

    for jj in range(0, nits):

        # Check for data spike:
        if np.count_nonzero(tau[:, jj, :]) < (co_array_num - 2):
            # We have a data spike, so do not process.
            continue

        # Now use original arrays
        y_var = tau[:, jj, :]

        residuals = y_var - (
            X_var @ coeffs[:, jj].reshape(dimension_number, 1))
        sor = np.sort(residuals.flatten()**2)
        s0 = np.sqrt(np.sum(sor[0:h]) / h) * raw_factor

        if np.abs(s0) < 1e-7:
            weights = (np.abs(residuals) < 1e-7)
            z_final = coeffs[:, jj].reshape(dimension_number, 1)
        else:
            weights = (np.abs(residuals / s0) <= quantile)
            weights = weights.flatten()
            # Cast logical to int
            weights_int = weights * 1

            # Perform the weighted least squares fit with
            # only data points with weight = 1
            # to increase statistical efficiency.
            q, r = np.linalg.qr(X_var[weights, :])
            qt = q.conj().T @ y_var[weights]
            z_final = np.linalg.lstsq(r, qt)[0]

            # Find dropped data points
            # Final residuals
            residuals = y_var - (X_var @ z_final)
            weights_num = np.sum(weights_int)
            scale = np.sqrt(np.sum(residuals[weights]**2) / (weights_num - 1))
            scale *= rew_factor1
            if weights_num != co_array_num:
                # Final fit - correction factor to make LTS approximately normally distributed # noqa
                rew_factor2 = rew_consfactor_lts(weights, dimension_number, co_array_num) # noqa
                scale *= rew_factor2
            weights = np.abs(residuals / scale) <= 2.5
            weights = weights.flatten()

        # Trace velocity & back-azimuth conversion
        # x-component of slowness vector
        sx = z_final[0][0]
        # y-component of slowness vector
        sy = z_final[1][0]
        # Calculate trace velocity from slowness
        lts_vel[jj] = 1/np.linalg.norm(z_final, 2)
        # Convert baz from mathematical CCW from E
        # to geographical CW from N. baz = arctan(sx/sy)
        lts_baz[jj] = (np.arctan2(sx, sy) * 180 / np.pi - 360) % 360

        # Uncertainty Quantification - Szuberla & Olson, 2004
        # Compute co-array eigendecomp. for uncertainty calcs.
        c_eig_vals, c_eig_vecs = np.linalg.eigh(xij[weights, :].T @ xij[weights, :])
        eig_vec_ang = np.arctan2(c_eig_vecs[1, 0], c_eig_vecs[0, 0])
        R = np.array([[np.cos(eig_vec_ang), np.sin(eig_vec_ang)],
                      [-np.sin(eig_vec_ang), np.cos(eig_vec_ang)]])

        # Calculate the sigma_tau value (Szuberla et al. 2006).
        residuals = tau[weights, jj, :] - (xij[weights, :] @ z_final)
        m_w, _ = np.shape(xij[weights, :])
        with np.errstate(invalid='raise'):
            try:
                sigma_tau[jj] = np.sqrt(tau[weights, jj, :].T @ residuals / (
                    m_w - dimension_number))[0]
            except FloatingPointError:
                pass

        # Equation 16 (Szuberla & Olson, 2004)
        sigS = sigma_tau[jj] / np.sqrt(c_eig_vals)
        # Form uncertainty ellipse major/minor axes
        a = np.sqrt(chi2) * sigS[0]
        b = np.sqrt(chi2) * sigS[1]
        # Rotate uncertainty ellipse to align major/minor axes
        # along coordinate system axes
        So = R @ [sx, sy]
        # Find angle & slowness extrema
        try:
            eExtrm, eVec = rthEllipse(a, b, So[0], So[1])
        except ValueError:
            eExtrm = np.array([np.nan, np.nan, np.nan, np.nan])
            eVec = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
        # Rotate eigenvectors back to original orientation
        eVec = eVec @ R
        # Fix up angle calculations
        sig_theta = np.abs(np.diff(
            (np.arctan2(eVec[2:, 1], eVec[2:, 0]) * 180 / np.pi - 360)
            % 360))
        if sig_theta > 180:
            sig_theta = np.abs(sig_theta - 360)

        # Halving here s.t. +/- value expresses uncertainty bounds.
        # Remove the 1/2's to get full values to express
        # coverage ellipse area.
        conf_int_baz[jj] = 0.5 * sig_theta
        conf_int_vel[jj] = 0.5 * np.abs(np.diff(1 / eExtrm[:2]))

        # Cast weights to int for output
        element_weights[:, jj] = weights * 1

    return lts_vel, lts_baz, element_weights, sigma_tau, conf_int_vel, conf_int_baz


def array_from_weights(weightarray, idx):
    """ Return array element pairs from LTS weights.

    Args:
        weightarray (array): An m x 0 array of the
            final LTS weights for each element pair.
        idx (array): An m x 2 array of the element pairs;
            generated from the `get_cc_time` function.

    Returns:
        (array):
        ``fstations``: A 1 x m array of element pairs.

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


##################
# Class definitions
##################
class LsBeam:
    """ Base class for least squares beamforming. This class is not meant to be used directly. """

    def __init__(self, data):
        # 2D Beamforming (trace_velocity and back-azimuth)
        self.dimension_number = 2
        # Pre-allocate Arrays
        # Median of the cross-correlation maxima
        self.mdccm = np.full(data.nits, np.nan)
        # Time
        self.t = np.full(data.nits, np.nan)
        # Trace Velocity [m/s]
        self.lts_vel = np.full(data.nits, np.nan)
        # Back-azimuth [degrees]
        self.lts_baz = np.full(data.nits, np.nan)
        # Calculate co-array and indices
        self.calculate_co_array(data)
        # Co-array size is N choose 2, N = num. of array elements
        self.co_array_num = int((data.nchans * (data.nchans - 1)) / 2)
        # Pre-allocate time delays
        self.tau = np.empty((self.co_array_num, data.nits))
        # Pre-allocate for median-absolute devation (MAD) of time delays
        self.time_delay_mad = np.zeros(data.nits)
        # Confidence interval for trace velocity
        self.conf_int_vel = np.full(data.nits, np.nan)
        # Confidence interval for back-azimuth
        self.conf_int_baz = np.full(data.nits, np.nan)
        # Pre-allocate for sigma-tau
        self.sigma_tau = np.full(data.nits, np.nan)
        # Specify station dictionary to maintain cross-compatibility
        self.stdict = {}
        # Confidence value for uncertainty calculation
        self.p = 0.90
        # Check co-array rank for least squares problem
        if (np.linalg.matrix_rank(self.xij) < self.dimension_number):
            raise RuntimeError('Co-array is ill posed for the least squares problem. Check array coordinates. xij rank < ' + str(self.dimension_number))

    def calculate_co_array(self, data):
        """ Calculate the co-array coordinates (x, y) for the array.
        """
        # Calculate element pair indices
        self.idx_pair = [(ii, jj) for ii in range(data.nchans - 1) for jj in range(ii + 1, data.nchans)] # noqa
        # Calculate the co-array
        self.xij = data.rij[:, np.array([ii[0] for ii in self.idx_pair])] - data.rij[:, np.array([jj[1] for jj in self.idx_pair])] # noqa
        self.xij = self.xij.T
        # Calculate median absolute deviation and standardized
        # co-array coordinates for least squares fit.
        self.xij_standardized = np.zeros_like(self.xij)
        self.xij_mad = np.zeros(2)
        for jj in range(0, self.dimension_number):
            self.xij_mad[jj] = 1.4826 * np.median(np.abs(self.xij[:, jj]))
            self.xij_standardized[:, jj] = self.xij[:, jj] / self.xij_mad[jj]

    def correlate(self, data):
        """ Cross correlate the time series data.
        """
        for jj in range(0, data.nits):
            # Get time from middle of window, except for the end.
            t0_ind = data.intervals[jj]
            tf_ind = data.intervals[jj] + data.winlensamp
            try:
                self.t[jj] = data.tvec[t0_ind + int(data.winlensamp/2)]
            except:
                self.t[jj] = np.nanmax(self.t, axis=0)

            # Numba doesn't accept mode='full' in np.correlate currently
            # Cross correlate the wave forms. Get the differential times.
            # Pre-allocate the cross-correlation matrix
            self.cij = np.empty((data.winlensamp * 2 - 1, self.co_array_num))
            for k in range(self.co_array_num):
                # MATLAB's xcorr w/ 'coeff' normalization:
                # unit auto-correlations.
                self.cij[:, k] = (np.correlate(data.data[t0_ind:tf_ind, self.idx_pair[k][0]], data.data[t0_ind:tf_ind, self.idx_pair[k][1]], mode='full') / np.sqrt(np.sum(data.data[t0_ind:tf_ind, self.idx_pair[k][0]] * data.data[t0_ind:tf_ind, self.idx_pair[k][0]]) * np.sum(data.data[t0_ind:tf_ind, self.idx_pair[k][1]] * data.data[t0_ind:tf_ind, self.idx_pair[k][1]]))) # noqa
            # Find the median of the cross-correlation maxima
            self.mdccm[jj] = np.nanmedian(self.cij.max(axis=0))
            # Form the time delay vector and save it
            delay = np.argmax(self.cij, axis=0) + 1
            self.tau[:, jj] = (data.winlensamp - delay) / data.sampling_rate
            self.time_delay_mad[jj] = 1.4826 * np.median(
                np.abs(self.tau[:, jj]))

        self.tau = np.reshape(self.tau, (self.co_array_num, data.nits, 1))


class OLSEstimator(LsBeam):
    """ Class for ordinary least squares beamforming."""

    def __init__(self, data):
        super().__init__(data)
        # Pre-compute co-array QR factorization for least squares
        self.q_xij, self.r_xij = np.linalg.qr(self.xij_standardized)

    def solve(self, data):
        """ Calculate trace velocity, back-azimuth, MdCCM, and confidence intervals.

        Args:
            data (DataBin): The DataBin object.
        """

        # Pre-compute co-array eigendecomp. for uncertainty calcs.
        c_eig_vals, c_eig_vecs = np.linalg.eigh(self.xij.T @ self.xij)
        eig_vec_ang = np.arctan2(c_eig_vecs[1, 0], c_eig_vecs[0, 0])
        R = np.array([[np.cos(eig_vec_ang), np.sin(eig_vec_ang)],
                      [-np.sin(eig_vec_ang), np.cos(eig_vec_ang)]])
        # Chi^2; default is 90% confidence (p = 0.90)
        # Special closed form for 2 degrees of freedom
        chi2 = -2 * np.log(1 - self.p)

        # Loop through time
        for jj in range(data.nits):

            # Check for data spike.
            if (self.time_delay_mad[jj] == 0) or (np.count_nonzero(self.tau[:, jj, :]) < (self.co_array_num - 2)):
                # We have a data spike, so do not process.
                continue

            y_var = self.tau[:, jj, :] / self.time_delay_mad[jj]
            qt = self.q_xij.conj().T @ y_var
            z_final = lstsq(self.r_xij, qt)[0]

            # Correct coefficients from standardization
            for ii in range(0, self.dimension_number):
                z_final[ii] *= self.time_delay_mad[jj] / self.xij_mad[ii] # noqa

            # x-component of slowness vector
            sx = z_final[0][0]
            # y-component of slowness vector
            sy = z_final[1][0]
            # Calculate trace velocity from slowness
            self.lts_vel[jj] = 1/np.linalg.norm(z_final, 2)
            # Convert baz from mathematical CCW from E
            # to geographical CW from N. baz = arctan(sx/sy)
            self.lts_baz[jj] = (np.arctan2(sx, sy) * 180 / np.pi - 360) % 360

            # Calculate the sigma_tau value (Szuberla et al. 2006).
            residuals = self.tau[:, jj, :] - (self.xij @ z_final)
            self.sigma_tau[jj] = np.sqrt(self.tau[:, jj, :].T @ residuals / (
                self.co_array_num - self.dimension_number))[0]

            # Calculate uncertainties from Szuberla & Olson, 2004
            # Equation 16
            sigS = self.sigma_tau[jj] / np.sqrt(c_eig_vals)
            # Form uncertainty ellipse major/minor axes
            a = np.sqrt(chi2) * sigS[0]
            b = np.sqrt(chi2) * sigS[1]
            # Rotate uncertainty ellipse to align major/minor axes
            # along coordinate system axes
            So = R @ [sx, sy]
            # Find angle & slowness extrema
            try:
                # rthEllipse routine can be unstable; catch instabilities
                eExtrm, eVec = rthEllipse(a, b, So[0], So[1])
                # Rotate eigenvectors back to original orientation
                eVec = eVec @ R
                # Fix up angle calculations
                sig_theta = np.abs(np.diff(
                    (np.arctan2(eVec[2:, 1], eVec[2:, 0]) * 180 / np.pi - 360)
                    % 360))
                if sig_theta > 180:
                    sig_theta = np.abs(sig_theta - 360)

                # Halving here s.t. +/- value expresses uncertainty bounds.
                # Remove the 1/2's to get full values to express
                # coverage ellipse area.
                self.conf_int_baz[jj] = 0.5 * sig_theta
                self.conf_int_vel[jj] = 0.5 * np.abs(np.diff(1 / eExtrm[:2]))

            except ValueError:
                self.conf_int_baz[jj] = np.nan
                self.conf_int_vel[jj] = np.nan


class LTSEstimator(LsBeam):
    """ Class for least trimmed squares (LTS) beamforming.
    """

    def __init__(self, data):
        super().__init__(data)
        # Pre-allocate array of slowness coefficients
        self.slowness_coeffs = np.empty((self.dimension_number, data.nits))
        # Pre-allocate weights
        self.element_weights = np.zeros((self.co_array_num, data.nits))
        # Raise error if a LTS object is instantiated with ALPHA = 1.0.
        # The ordinary least squares code should be used instead
        if data.alpha == 1.0:
            raise RuntimeError('ALPHA = 1.0. This class is computionally inefficient. Use the OLSEstimator class instead.')
        # Raise error if there are too few data points for subsetting
        if np.shape(self.xij)[0] < (2 * self.dimension_number):
            raise RuntimeError('The co-array must have at least 4 elements for least trimmed squares. Check rij array coordinates.')
        # Calculate the subset size.
        self.h_calc(data)
        # The number of subsets we will test.
        self.n_samples = 500
        # The number of best subsets to try in the final iteration.
        self.candidate_size = 10
        # The initial number of concentration steps.
        self.csteps = 4
        # The number of concentration steps for the second stage.
        self. csteps2 = 100

    def h_calc(self, data):
        r""" Generate the h-value, the number of points to fit.

        Args:
            ALPHA (float): The decimal percentage of points
                to keep. Default is 0.75.
            n (int): The total number of points.
            p (int): The number of parameters.

        Returns:
            (int):
            ``h``: The number of points to fit.
        """

        self.h = int(np.floor(2*np.floor((self.co_array_num + self.dimension_number + 1) / 2) - self.co_array_num + 2 * (self.co_array_num - np.floor((self.co_array_num + self.dimension_number + 1) / 2)) * data.alpha)) # noqa

    def solve(self, data):
        """ Apply the FAST_LTS algorithm to calculate a least trimmed squares solution for trace velocity, back-azimuth, MdCCM, and confidence intervals.

        Args:
            data (DataBin): The DataBin object.
        """
        # Determine the best slowness coefficients from FAST-LTS
        self.slowness_coeffs = fast_LTS(data.nits, self.tau, self.time_delay_mad, self.xij_standardized, self.xij_mad, self.dimension_number, self.candidate_size, self.n_samples, self.co_array_num, self.slowness_coeffs, self.csteps, self.h, self.csteps2, random_set, check_array) # noqa
        # Use the best slowness coefficients to determine dropped stations
        # Calculate uncertainties at 90% confidence
        self.lts_vel, self.lts_baz, self.element_weights, self.sigma_tau, self.conf_int_vel, self.conf_int_baz = post_process(self.dimension_number, self.co_array_num, data.alpha, self.h, data.nits, self.tau, self.xij, self.slowness_coeffs, self.lts_vel, self.lts_baz, self.element_weights, self.sigma_tau, self.p, self.conf_int_vel, self.conf_int_baz) # noqa
        # Find dropped stations from weights
        # Map dropped data points back to elements.
        for jj in range(0, data.nits):
            stns = array_from_weights(
                self.element_weights[:, jj], self.idx_pair)
            # Stash the number of elements for plotting.
            if len(stns) > 0:
                tval = str(self.t[jj])
                self.stdict[tval] = stns
            if jj == (data.nits - 1) and data.alpha != 1.0:
                self.stdict['size'] = data.nchans
