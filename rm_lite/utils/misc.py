#!/usr/bin/env python3
import warnings
from typing import Callable, Literal, NamedTuple, Optional, Tuple

from astropy.stats import akaike_info_criterion_lsq
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
from scipy.stats import norm
from scipy.optimize import curve_fit

from uncertainties import unumpy


class FitResult(NamedTuple):
    popt: np.ndarray
    pcov: np.ndarray
    stokes_i_model_func: Callable
    aic: float
    stokes_i_model_array: np.ndarray


# -----------------------------------------------------------------------------#
def calc_mom2_FDF(FDF, phiArr):
    """
    Calculate the 2nd moment of the polarised intensity FDF. Can be applied to
    a clean component spectrum or a standard FDF
    """

    K = np.sum(np.abs(FDF))
    phiMean = np.sum(phiArr * np.abs(FDF)) / K
    phiMom2 = np.sqrt(np.sum(np.power((phiArr - phiMean), 2.0) * np.abs(FDF)) / K)

    return phiMom2


# -----------------------------------------------------------------------------#
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    """
    Calculate the vertex of a parabola given three adjacent points.
    Normalization of coordinates must be performed first to reduce risk of
    floating point errors.
    """
    midpoint = x2
    deltax = x2 - x3
    yscale = y2
    (x1, x2, x3) = [(x - x2) / deltax for x in (x1, x2, x3)]  # slide spectrum to zero
    (y1, y2, y3) = [y / yscale for y in (y1, y2, y3)]

    D = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / D
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / D
    C = (
        x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3
    ) / D

    xv = -B / (2.0 * A)
    yv = C - B * B / (4.0 * A)

    return xv * deltax + midpoint, yv * yscale


def renormalize_StokesI_model(
    fit_result: FitResult, new_reference_frequency: float
) -> FitResult:
    """Adjust the reference frequency for the Stokes I model and fix the fit
    parameters such that the the model is the same.

    This is important because the initial Stokes I fitted model uses an arbitrary
    reference frequency, and it may be desirable for users to know the exact
    reference frequency of the model.

    This function now includes the ability to transform the model parameter
    errors to the new reference frequency. This feature uses a first order
    approximation, that scales with the ratio of new to old reference frequencies.
    Large changes in reference frequency may be outside the linear valid regime
    of the first order approximation, and thus should be avoided.

    Args:
        fit_result (FitResult): the result of a Stokes I model fit.
        new_reference_frequency (float): the new reference frequency for the model.

    Returns:
        FitResult: the fit results with the reference frequency adjusted to the new value.


    """
    # Renormalization ratio:
    x = new_reference_frequency / fit_result.reference_frequency_Hz

    # Check if ratio is within zone of probable accuracy (approx. 10%, from empirical tests)
    if (x < 0.9) or (x > 1.1):
        warnings.warn(
            "New Stokes I reference frequency more than 10% different than original, uncertainties may be unreliable",
            UserWarning,
        )

    (a, b, c, d, f, g) = fit_result.params

    # Modify fit parameters to new reference frequency.
    # I have derived all these conversion equations analytically for the
    # linear- and log-polynomial models.
    if fit_result.fit_function == "linear":
        new_parms = [a * x**5, b * x**4, c * x**3, d * x**2, f * x, g]
    elif fit_result.fit_function == "log":
        lnx = np.log10(x)
        new_parms = [
            a,
            5 * a * lnx + b,
            10 * a * lnx**2 + 4 * b * lnx + c,
            10 * a * lnx**3 + 6 * b * lnx**2 + 3 * c * lnx + d,
            5 * a * lnx**4 + 4 * b * lnx**3 + 3 * c * lnx**2 + 2 * d * lnx + f,
            g
            * np.power(10, a * lnx**5 + b * lnx**4 + c * lnx**3 + d * lnx**2 + f * lnx),
        ]

    # Modify fit parameter errors to new reference frequency.
    # Note this implicitly makes a first-order approximation in the correletion
    # structure between uncertainties
    # The general equation for the transformation of uncertainties is:
    #   var(p) = sum_i,j((\partial p / \partial a_i) * (\partial p / \partial a_j) * cov(a_i,a_j))
    # where a_i are the initial parameters, p is a final parameter,
    # and the partial derivatives are evaluated at the fit parameter values (and frequency ratio).
    # The partial derivatives all come from the parameter conversion equations above.

    cov = fit_result.pcov
    if fit_result.fit_function == "linear":
        new_errors = [
            np.sqrt(x**10 * cov[0, 0]),
            np.sqrt(x**8 * cov[1, 1]),
            np.sqrt(x**6 * cov[2, 2]),
            np.sqrt(x**4 * cov[3, 3]),
            np.sqrt(x**2 * cov[4, 4]),
            np.sqrt(cov[5, 5]),
        ]
    elif fit_result.fit_function == "log":
        g2 = new_parms[5]  # Convenient shorthand for new value of g variable.
        new_errors = [
            np.sqrt(cov[0, 0]),
            np.sqrt(25 * lnx**2 * cov[0, 0] + 10 * lnx * cov[0, 1] + cov[1, 1]),
            np.sqrt(
                100 * lnx**4 * cov[0, 0]
                + 80 * lnx**3 * cov[0, 1]
                + 20 * lnx**2 * cov[0, 2]
                + 16 * lnx**2 * cov[1, 1]
                + 8 * lnx * cov[1, 2]
                + cov[2, 2]
            ),
            np.sqrt(
                100 * lnx**6 * cov[0, 0]
                + 120 * lnx**5 * cov[0, 1]
                + 60 * lnx**4 * cov[0, 2]
                + 20 * lnx**3 * cov[0, 3]
                + 36 * lnx**4 * cov[1, 1]
                + 36 * lnx**3 * cov[1, 2]
                + 12 * lnx**2 * cov[1, 3]
                + 9 * lnx**2 * cov[2, 2]
                + 6 * lnx * cov[2, 3]
                + cov[3, 3]
            ),
            np.sqrt(
                25 * lnx**8 * cov[0, 0]
                + 40 * lnx**7 * cov[0, 1]
                + 30 * lnx**6 * cov[0, 2]
                + 20 * lnx**5 * cov[0, 3]
                + 10 * lnx**4 * cov[0, 4]
                + 16 * lnx**6 * cov[0, 5]
                + 24 * lnx**5 * cov[1, 2]
                + 16 * lnx**4 * cov[1, 3]
                + 8 * lnx**3 * cov[1, 4]
                + 9 * lnx**4 * cov[2, 2]
                + 12 * lnx**3 * cov[2, 3]
                + 6 * lnx**2 * cov[2, 4]
                + 4 * lnx**2 * cov[3, 3]
                + 4 * lnx * cov[3, 4]
                + cov[4, 4]
            ),
            np.abs(g2)
            * np.sqrt(
                lnx**10 * cov[0, 0]
                + 2 * lnx**9 * cov[0, 1]
                + 2 * lnx**8 * cov[0, 2]
                + 2 * lnx**7 * cov[0, 3]
                + 2 * lnx**6 * cov[0, 4]
                + 2 * lnx**5 / g * np.log(10) * cov[0, 5]
                + lnx**8 * cov[1, 1]
                + 2 * lnx**7 * cov[1, 2]
                + 2 * lnx**6 * cov[1, 3]
                + 2 * lnx**5 * cov[1, 4]
                + 2 * lnx**4 / g * np.log(10) * cov[1, 5]
                + lnx**6 * cov[2, 2]
                + 2 * lnx**5 * cov[2, 3]
                + 2 * lnx**4 * cov[2, 4]
                + 2 * lnx**3 / g * np.log(10) * cov[2, 5]
                + lnx**4 * cov[3, 3]
                + 2 * lnx**3 * cov[3, 4]
                + 2 * lnx**2 / g * np.log(10) * cov[3, 5]
                + lnx**2 * cov[4, 4]
                + 2 * lnx / g * np.log(10) * cov[4, 5]
                + 1 / g**2 * cov[5, 5]
            ),
        ]

    return fit_result.with_options(
        params=new_parms,
        reference_frequency_Hz=new_reference_frequency,
        perror=new_errors,
    )


def polynomial(order: int) -> Callable:
    def poly_func(x: np.ndarray, *params) -> np.ndarray:
        result = 0
        for i in range(order + 1):
            result += params[i] * x**i
        return result

    return poly_func


def power_law(order: int) -> Callable:
    def power_func(x: np.ndarray, *params) -> np.ndarray:
        power = 0

        for i in range(order + 1):
            power += params[i] * np.log10(x**i)
        return x**power

    return power_func


def best_aic_func(aics: np.ndarray, n_param: np.ndarray) -> Tuple[float, int, int]:
    """Find the best AIC for a set of AICs using Occam's razor."""
    # Find the best AIC
    best_aic_idx = int(np.nanargmin(aics))
    best_aic = float(aics[best_aic_idx])
    best_n = int(n_param[best_aic_idx])
    # logger.debug(f"Lowest AIC is {best_aic}, with {best_n} params.")
    # Check if lower have diff < 2 in AIC
    aic_abs_diff = np.abs(aics - best_aic)
    bool_min_idx = np.zeros_like(aics).astype(bool)
    bool_min_idx[best_aic_idx] = True
    potential_idx = (aic_abs_diff[~bool_min_idx] < 2) & (
        n_param[~bool_min_idx] < best_n
    )
    if not any(potential_idx):
        return best_aic, best_n, best_aic_idx

    bestest_n = int(np.min(n_param[~bool_min_idx][potential_idx]))
    bestest_aic_idx = int(np.where(n_param == bestest_n)[0][0])
    bestest_aic = float(aics[bestest_aic_idx])
    # logger.debug(
    #     f"Model within 2 of lowest AIC found. Occam says to take AIC of {bestest_aic}, with {bestest_n} params."
    # )
    return bestest_aic, bestest_n, bestest_aic_idx


def static_fit(
    freq_array_hz: np.ndarray,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_type == "linear":
        fit_func = polynomial(fit_order)
    elif fit_type == "log":
        fit_func = power_law(fit_order)

    popt, pcov = curve_fit(
        fit_func,
        freq_array_hz,
        stokes_i_array,
        sigma=stokes_i_error_array,
        absolute_sigma=True,
        p0=np.zeros(fit_order + 1),
    )
    stokes_i_model_array = fit_func(freq_array_hz, *popt)
    ssr = np.sum((stokes_i_array - stokes_i_model_array) ** 2)
    aic = akaike_info_criterion_lsq(
        ssr=ssr, n_params=fit_order + 1, n_samples=len(freq_array_hz)
    )

    return FitResult(
        popt=popt,
        pcov=pcov,
        stokes_i_model_func=fit_func,
        aic=aic,
        stokes_i_model_array=stokes_i_model_array,
    )


def dynamic_fit(
    freq_array_hz: np.ndarray,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    orders = np.arange(1, fit_order + 1)
    fit_results = []

    for i, order in enumerate(orders):
        fit_result = static_fit(
            freq_array_hz,
            stokes_i_array,
            stokes_i_error_array,
            order,
            fit_type,
        )
        fit_results.append(fit_result)

    aics = np.array([fit_result.aic for fit_result in fit_results])
    bestest_aic, bestest_n, bestest_aic_idx = best_aic_func(aics, orders)

    return fit_results[bestest_aic_idx]


def fit_stokes_i_model(
    freq_array_hz: np.ndarray,
    stokes_i_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    fit_order: int = 2,
    fit_type: Literal["log", "linear"] = "log",
) -> FitResult:
    if fit_order < 0:
        return dynamic_fit(
            freq_array_hz,
            stokes_i_array,
            stokes_i_error_array,
            abs(fit_order),
            fit_type,
        )

    return static_fit(
        freq_array_hz,
        stokes_i_array,
        stokes_i_error_array,
        fit_order,
        fit_type,
    )


class FractionalSpectra(NamedTuple):
    stokes_i_model_array: np.ndarray
    stokes_q_frac_array: np.ndarray
    stokes_u_frac_array: np.ndarray
    stokes_q_frac_error_array: np.ndarray
    stokes_u_frac_error_array: np.ndarray


def create_fractional_spectra(
    freq_array_hz: np.ndarray,
    stokes_i_array: np.ndarray,
    stokes_q_array: np.ndarray,
    stokes_u_array: np.ndarray,
    stokes_i_error_array: np.ndarray,
    stokes_q_error_array: np.ndarray,
    stokes_u_error_array: np.ndarray,
    fit_order: int = 2,
    fit_function: Literal["log", "linear"] = "log",
    stokes_i_model_array: Optional[np.ndarray] = None,
) -> FractionalSpectra:
    no_nan_idx = (
        np.isfinite(stokes_i_array)
        & np.isfinite(stokes_i_error_array)
        & np.isfinite(stokes_q_array)
        & np.isfinite(stokes_q_error_array)
        & np.isfinite(stokes_u_array)
        & np.isfinite(stokes_u_error_array)
        & np.isfinite(freq_array_hz)
    )

    freq_array_hz = freq_array_hz[no_nan_idx]
    stokes_i_array = stokes_i_array[no_nan_idx]
    stokes_q_array = stokes_q_array[no_nan_idx]
    stokes_u_array = stokes_u_array[no_nan_idx]
    stokes_i_error_array = stokes_i_error_array[no_nan_idx]
    stokes_q_error_array = stokes_q_error_array[no_nan_idx]
    stokes_u_error_array = stokes_u_error_array[no_nan_idx]

    stokes_i_uarray = unumpy.uarray(stokes_i_array, stokes_i_error_array)
    stokes_q_uarray = unumpy.uarray(stokes_q_array, stokes_q_error_array)
    stokes_u_uarray = unumpy.uarray(stokes_u_array, stokes_u_error_array)
    if stokes_i_model_array is not None:
        stokes_i_model_uarray = unumpy.uarray(stokes_i_model_array, 0.0)
    else:
        fit_result = fit_stokes_i_model(
            freq_array_hz,
            stokes_i_array,
            stokes_i_error_array,
            fit_order,
            fit_function,
        )

    stokes_q_frac_uarray = stokes_q_uarray / stokes_i_model_uarray
    stokes_u_frac_uarray = stokes_u_uarray / stokes_i_model_uarray

    stokes_q_frac_array = unumpy.nominal_values(stokes_q_frac_uarray)
    stokes_u_frac_array = unumpy.nominal_values(stokes_u_frac_uarray)
    stokes_q_frac_error_array = unumpy.std_devs(stokes_q_frac_uarray)
    stokes_u_frac_error_array = unumpy.std_devs(stokes_u_frac_uarray)

    return FractionalSpectra(
        stokes_i_model_array=stokes_i_model_array,
        stokes_q_frac_array=stokes_q_frac_array,
        stokes_u_frac_array=stokes_u_frac_array,
        stokes_q_frac_error_array=stokes_q_frac_error_array,
        stokes_u_frac_error_array=stokes_u_frac_error_array,
    )


# -----------------------------------------------------------------------------#
def interp_images(arr1, arr2, f=0.5):
    """Create an interpolated image between two other images."""

    nY, nX = arr1.shape

    # Concatenate arrays into a single array of shape (2, nY, nX)
    arr = np.r_["0,3", arr1, arr2]

    # Define the grid coordinates where you want to interpolate
    X, Y = np.meshgrid(np.arange(nX), np.arange(nY))

    # Create coordinates for interpolated frame
    coords = np.ones(arr1.shape) * f, Y, X

    # Interpolate using the map_coordinates function
    interpArr = ndi.map_coordinates(arr, coords, order=1)

    return interpArr


# -----------------------------------------------------------------------------#
def fit_spec_poly5(xData, yData, dyData=None, order=5, fit_function="log"):
    """Fit a 5th order polynomial to a spectrum. To avoid overflow errors the
    X-axis data should not be large numbers (e.g.: x10^9 Hz; use GHz
    instead)."""

    # Impose limits on polynomial order
    if order < 0:
        order = np.abs(order)
    if order > 5:
        order = 5
    if dyData is None:
        dyData = np.ones_like(yData)
    if np.all(dyData == 0):
        dyData = np.ones_like(yData)

    # Estimate starting coefficients
    C1 = 0.0
    C0 = np.nanmean(yData)
    C5 = 0.0
    C4 = 0.0
    C3 = 0.0
    C2 = 0.0
    inParms = [
        {"value": C5, "parname": "C5", "fixed": False},
        {"value": C4, "parname": "C4", "fixed": False},
        {"value": C3, "parname": "C3", "fixed": False},
        {"value": C2, "parname": "C2", "fixed": False},
        {"value": C1, "parname": "C1", "fixed": False},
        {"value": C0, "parname": "C0", "fixed": False},
    ]

    # Set the parameters as fixed of > order
    for i in range(len(inParms)):
        if len(inParms) - i - 1 > order:
            inParms[i]["fixed"] = True

    # Function to evaluate the difference between the model and data.
    # This is minimised in the least-squared sense by the fitter
    if fit_function == "linear":

        def errFn(p, fjac=None):
            status = 0
            return status, (poly5(p)(xData) - yData) / dyData

    elif fit_function == "log":

        def errFn(p, fjac=None):
            status = 0
            return status, (powerlaw_poly5(p)(xData) - yData) / dyData

    # Use MPFIT to perform the LM-minimisation
    mp = mpfit(errFn, parinfo=inParms, quiet=True)

    return mp


# -----------------------------------------------------------------------------#


def poly5(p):
    """Returns a function to evaluate a polynomial. The subfunction can be
    accessed via 'argument unpacking' like so: 'y = poly5(p)(*x)',
    where x is a vector of X values and p is a vector of coefficients."""

    # Fill out the vector to length 6 if necessary
    p = np.append(np.zeros((6 - len(p))), p)

    def rfunc(x):
        y = (
            p[0] * x**5.0
            + p[1] * x**4.0
            + p[2] * x**3.0
            + p[3] * x**2.0
            + p[4] * x
            + p[5]
        )
        return y

    return rfunc


def powerlaw_poly5(p):
    """Returns a function to evaluate a power law polynomial. The subfunction can be
    accessed via 'argument unpacking' like so: 'y = powerlaw_poly5(p)(*x)',
    where x is a vector of X values and p is a vector of coefficients."""

    # Fill out the vector to length 6 if necessary
    p = np.append(np.zeros((6 - len(p))), p)

    def rfunc(x):
        y = (
            p[0] * np.log10(x) ** 4.0
            + p[1] * np.log10(x) ** 3.0
            + p[2] * np.log10(x) ** 2.0
            + p[3] * np.log10(x)
            + p[4]
        )
        return p[5] * np.power(x, y)

    return rfunc


# -----------------------------------------------------------------------------#
def nanmedian(arr, **kwargs):
    """
    Returns median ignoring NaNs.
    """

    return ma.median(ma.masked_where(arr != arr, arr), **kwargs)


# -----------------------------------------------------------------------------#
def nanmean(arr, **kwargs):
    """
    Returns mean ignoring NaNs.
    """

    return ma.mean(ma.masked_where(arr != arr, arr), **kwargs)


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def extrap(x, xp, yp):
    """
    Wrapper to allow np.interp to linearly extrapolate at function ends.

    np.interp function with linear extrapolation
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate
    -give-a-an-extrapolated-result-beyond-the-input-ran
    """

    y = np.interp(x, xp, yp)
    y = np.where(x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y)
    y = np.where(
        x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
    )
    return y


# -----------------------------------------------------------------------------#
def toscalar(a):
    """
    Returns a scalar version of a Numpy object.
    """
    try:
        return a.item()
    except Exception:
        return a


# -----------------------------------------------------------------------------#
def MAD(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:
    median(abs(a - median(a))) / c
    c = 0.6745 is the constant to convert from MAD to std
    """

    a = ma.masked_where(a != a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a, 0, axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)

    return m


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def create_pqu_spectra_burn(
    freqArr_Hz, fracPolArr, psi0Arr_deg, RMArr_radm2, sigmaRMArr_radm2=None
):
    """Return fractional P/I, Q/I & U/I spectra for a sum of Faraday thin
    components (multiple values may be given as a list for each argument).
    Burn-law external depolarisation may be applied to each
    component via the optional 'sigmaRMArr_radm2' argument. If
    sigmaRMArr_radm2=None, all values are set to zero, i.e., no
    depolarisation."""

    # Convert lists to arrays
    freqArr_Hz = np.array(freqArr_Hz, dtype="f8")
    fracPolArr = np.array(fracPolArr, dtype="f8")
    psi0Arr_deg = np.array(psi0Arr_deg, dtype="f8")
    RMArr_radm2 = np.array(RMArr_radm2, dtype="f8")
    if sigmaRMArr_radm2 is None:
        sigmaRMArr_radm2 = np.zeros_like(fracPolArr)
    else:
        sigmaRMArr_radm2 = np.array(sigmaRMArr_radm2, dtype="f8")

    # Calculate some prerequsites
    nChans = len(freqArr_Hz)
    nComps = len(fracPolArr)
    lamArr_m = C / freqArr_Hz
    lamSqArr_m2 = np.power(lamArr_m, 2.0)

    # Convert the inputs to column vectors
    fracPolArr = fracPolArr.reshape((nComps, 1))
    psi0Arr_deg = psi0Arr_deg.reshape((nComps, 1))
    RMArr_radm2 = RMArr_radm2.reshape((nComps, 1))
    sigmaRMArr_radm2 = sigmaRMArr_radm2.reshape((nComps, 1))

    # Calculate the p, q and u Spectra for all components
    pArr = fracPolArr * np.ones((nComps, nChans), dtype="f8")
    quArr = pArr * (
        np.exp(2j * (np.radians(psi0Arr_deg) + RMArr_radm2 * lamSqArr_m2))
        * np.exp(-2.0 * sigmaRMArr_radm2 * np.power(lamArr_m, 4.0))
    )

    # Sum along the component axis to create the final spectra
    quArr = quArr.sum(0)
    qArr = quArr.real
    uArr = quArr.imag
    pArr = np.abs(quArr)

    return pArr, qArr, uArr


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    """Return the CDF of a normal distribution between -6 and 6 sigma, or at
    the values of an input array."""

    if xArr is None:
        x = np.linspace(-6.0 * std, 6.0 * std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)

    return x, y
