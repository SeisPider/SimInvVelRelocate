# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 15:47h, 20/06/2018
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
# -------------------------------------------------------------------------------
"""Construct related utilization

References
==========
1. Brocher, Thomas M. "Empirical relations between elastic wavespeeds and density 
   in the Earth's crust." Bulletin of the seismological Society of America 
   95.6 (2005): 2081-2092.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize
from functools import partial
from patsy import dmatrix
from scipy.interpolate import interp1d
from numba import jit
import scipy
from numpy.fft import rfft, rfftfreq, irfft
#from interval import Interval


def seperate_channels(st, comps=["R", "T", "Z"]):
    """Seperate channels from obspy Stream obj.

    Parameters
    ==========
    st : obspy.Stream
        stream storing all three channels
    comps : list
        channels to be seperated, [RTZ] or [ENZ]
    """
    trs = []
    for comp in comps:
        trs.append(st.select(component=comp)[0])
    return tuple(trs)


def Quan_I(freq, AmpN, AmpE, minfreq, maxfreq, grid=0.1, min_ang=0, max_ang=360):
    """Find the RTZ coordinates at this particular time slip
    Parameter
    =========
    freq: numpy.array
        freqeuency range of spectrums
    AmpN: numpy.array
        spectrum of Northern trace
    AmpE: numpy.array
        spectrum of Eastern trace
    minfreq: float
        minimum frequency used to estimated I
    maxfreq: float
        maximum frequency used to estimated I
    grid: float
        degree interval in searching the rotation angle, in degree
    min_ang: float
        minimum rotation angle to start searching, in degree
    max_ang: float
        maximum rotation angle to end searching, in degree
    """
    # Get the search angles
    angles = np.deg2rad(np.arange(min_ang, max_ang, grid))

    # Trim spectrum based on given frequency band
    condition = (freq >= minfreq) * (freq <= maxfreq)
    msk = np.where(condition)
    TAmpN, TAmpE = AmpN[msk], AmpE[msk]

    # Grid search each possible rotation angle
    QuanI = np.array(
        [
            (np.abs(TAmpN * np.cos(ang) + TAmpE * np.sin(ang)) ** 2).sum()
            for ang in angles
        ]
    )
    # Seach angle that maximum the QuanI
    ang_max = angles[QuanI.argmax()]

    # give the rotated spectrum
    maxH = TAmpN * np.cos(ang_max) + TAmpE * np.sin(ang_max)
    minH = TAmpN * np.cos(ang_max + np.pi / 2) + TAmpE * np.sin(ang_max + np.pi / 2)
    return ang_max, maxH, minH, msk


def vp_rho_to_vs(vs, mode="both"):
    """Compute p wave speed and density as a function of s wave speed 
    according to eqs. 9 & in Ref. 1.

    Parameters
    ==========
    vs: numpy.array
        to be related vs
    
    Constrains
    ==========
    1. Vs belongs to [0, 4.5] km/s
    2. Vp belongs to [1.5, 7.72] km/s
    3. This mapping relations fits for crystallized crustal structure
    """
    # eqs. 9 with con. 1
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs ** 2 + 0.2683 * vs ** 3 - 0.0251 * vs ** 4
    if mode == "vpvs":
        return vp
    # eqs. 1 with con. 2
    rho = (
        1.6612 * vp
        - 0.4721 * vp ** 2
        + 0.0671 * vp ** 3
        - 0.0043 * vp ** 4
        + 0.000106 * vp ** 5
    )
    if mode == "rhovs":
        return rho

    if mode == "both":
        return vp, rho


def DFT(data, dt=1, f=0.2):
    """Compute FT formation of a time sery data at 
    specific freqeuency f

    Parameters
    ==========
    data: numpy.array
        time sery data
    dt: float
        time step of the time sery
    f: float
        specific frequency to compute FT 
    
    References
    ==========
    Eq.1 from https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    """
    # Angular frequency
    iomega = -1j * 2 * np.pi * f

    # Time series
    N = len(data)
    TN = np.arange(0, N) * dt

    # Phase part
    WN = (data * np.exp(iomega * TN)).sum()
    return WN


def moving_avg(a, halfwindow, mask=None):
    """
    Performs a fast n-point moving average of (the last
    dimension of) array *a*, by using stride tricks to roll
    a window on *a*.
    Note that *halfwindow* gives the nb of points on each side,
    so that n = 2*halfwindow + 1.
    If *mask* is provided, values of *a* where mask = False are
    skipped.
    Returns an array of same size as *a* (which means that near
    the edges, the averaging window is actually < *npt*).
    """
    # padding array with zeros on the left and on the right:
    # e.g., if halfwindow = 2:
    # a_padded    = [0 0 a0 a1 ... aN 0 0]
    # mask_padded = [F F ?  ?      ?  F F]

    if mask is None:
        mask = np.ones_like(a, dtype="bool")

    zeros = np.zeros(a.shape[:-1] + (halfwindow,))
    falses = zeros.astype("bool")

    a_padded = np.concatenate((zeros, np.where(mask, a, 0), zeros), axis=-1)
    mask_padded = np.concatenate((falses, mask, falses), axis=-1)

    # rolling window on padded array using stride trick
    #
    # E.g., if halfwindow=2:
    # rolling_a[:, 0] = [0   0 a0 a1 ...    aN]
    # rolling_a[:, 1] = [0  a0 a1 a2 ... aN 0 ]
    # ...
    # rolling_a[:, 4] = [a2 a3 ...    aN  0  0]

    npt = 2 * halfwindow + 1  # total size of the averaging window
    rolling_a = as_strided(
        a_padded, shape=a.shape + (npt,), strides=a_padded.strides + (a.strides[-1],)
    )
    rolling_mask = as_strided(
        mask_padded,
        shape=mask.shape + (npt,),
        strides=mask_padded.strides + (mask.strides[-1],),
    )

    # moving average
    n = rolling_mask.sum(axis=-1)
    return np.where(n > 0, rolling_a.sum(axis=-1).astype("float") / n, np.nan)


def time_normalization(
    tr, freqmin_earthquake=0.02, freqmax_earthquake=0.3, corners=2, zerophase=True
):
    # normalization of the signal by the running mean
    # in the earthquake frequency band
    tr.filter(
        type="bandpass",
        freqmin=freqmin_earthquake,
        freqmax=freqmax_earthquake,
        corners=corners,
        zerophase=zerophase,
    )
    # Time-normalization weights from smoothed abs(data)
    # Note that trace's data can be a masked array
    window_time = 1 / (2 * freqmin_earthquake)
    halfwindow = int(round(window_time * tr.stats.sampling_rate / 2))
    mask = ~tr.data.mask if np.ma.isMA(tr.data) else None
    tnorm_w = moving_avg(np.abs(tr.data), halfwindow=halfwindow, mask=mask)
    if np.ma.isMA(tr.data):
        # turning time-normalization weights into a masked array
        tnorm_w = np.ma.masked_array(tnorm_w, tr.data.mask)
    if np.any((tnorm_w == 0.0) | np.isnan(tnorm_w)):
        # illegal normalizing value -> skipping trace
        raise ValueError("Zero or NaN normalization weight")
    # time-normalization
    tr.data /= tnorm_w
    return tr.data


def get_fill(st, starttime=None, endtime=None):
    """
    Subroutine to get data fill
    @rtype: float
    """
    if len(st) == 0:
        # no trace
        return 0.0

    ststart = min(tr.stats.starttime for tr in st)
    stend = max(tr.stats.endtime for tr in st)
    dttot = (stend if not endtime else endtime) - (
        ststart if not starttime else starttime
    )
    gaps = st.get_gaps()

    fill = 1.0
    if starttime:
        fill -= max(ststart - starttime, 0.0) / dttot
    if endtime:
        fill -= max(endtime - stend, 0.0) / dttot

    for g in gaps:
        gapstart = g[4]
        gapend = g[5]
        if starttime:
            gapstart = max(gapstart, starttime)
            gapend = max(gapend, starttime)
        if endtime:
            gapstart = min(gapstart, endtime)
            gapend = min(gapend, endtime)
        fill -= (gapend - gapstart) / dttot

    return fill


def get_nonmask_fill(st, starttime=None, endtime=None):
    """Get the persontage of masked data points
    """
    if len(st) == 0:
        # no trace
        return 0.0

    maskptnum, allptnum = 0.0, 0.0
    for tr in st:
        # time mask
        dt = 1.0 / tr.stats.sampling_rate
        timescale = np.arange(tr.stats.starttime, tr.stats.endtime, dt)
        condition = (timescale <= endtime) * (timescale >= starttime)
        msk = np.where(condition)

        # Get points
        try:
            dataarray, maskarray = tr.data.data, tr.data.mask
            maskptnum += len(dataarray[msk][maskarray[msk]])
            allptnum += int((endtime - starttime) / dt)
        except AttributeError:
            allptnum += int((endtime - starttime) / dt)
    return float(1 - maskptnum / allptnum)


def fit_value_with_spline_basis(basisx, value, datax):
    """fit a profile with spline basis
    """
    initmod = np.ones(6) * value.mean()
    splineBasis = dmatrix(
        "bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": basisx}
    )

    part_misfit = partial(
        fit_value_misfit,
        splineBasis=splineBasis,
        basisx=basisx,
        value=value,
        datax=datax,
    )
    vmax = np.abs(value).max()
    vmin = -1.0 * vmax
    options = {"maxiter": 1000, "ftol": 1e-8}
    bnds = (
        (vmin, vmax),
        (vmin, vmax),
        (vmin, vmax),
        (vmin, vmax),
        (vmin, vmax),
        (vmin, vmax),
    )
    x = minimize(part_misfit, x0=initmod, method="SLSQP", bounds=bnds, options=options)
    return x


def fit_value_misfit(mod, splineBasis, basisx, value, datax):
    """misfit to fit the velocity
    """
    vmod = syn_value(mod, splineBasis, basisx, datax)
    return (np.abs(vmod - value)).sum()


def syn_value(coef, splineBasis, basisx, datax):
    """Compute the composed y with given spline function coefficients on x 
    """
    # generate the basis function
    tmp = np.dot(splineBasis, coef)

    # interpolation to obtain result at the datax
    fx = interp1d(basisx, tmp, kind="linear", fill_value="extrapolate")
    vmod = fx(datax)
    return vmod


@jit(nopython=True)
def welch_window(freqs, lft_bound, rgt_bound):
    """define the welch window used for constructure filter
    """
    # Compute decay function based on the welch window
    responses = []
    center_f = (lft_bound + rgt_bound) / 2
    half_df = (rgt_bound - lft_bound) / 2
    for freq in freqs:
        if freq <= lft_bound or freq >= rgt_bound:
            responses.append(0)
        else:
            response = 1 - ((freq - center_f) / half_df) ** 2
            responses.append(response)
    return np.array(responses)


@jit(nopython=True)
def unwrap_cycles(ref_prop_phase, syn_phase, obs_phase):
    """same as name
    """
    cum_cycle_num = np.floor(ref_prop_phase / (np.pi * 2))
    syn_prop_phase = syn_phase + cum_cycle_num * np.pi * 2 + np.pi
    obs_prop_phase = obs_phase + cum_cycle_num * np.pi * 2 + np.pi
    return syn_prop_phase, obs_prop_phase


@jit(nopython=True)
def estimate_possible_vc_pert(refc, dist, freqs, obs_prop_phase, syn_prop_phase):
    """same as name
    """
    dpsi = obs_prop_phase - syn_prop_phase
    dc = dpsi * refc ** 2 / (-2.0 * np.pi * freqs * dist)
    return dc


def resample(trace, dt_resample):
    """
    Subroutine to resample trace

    @type trace: L{obspy.core.trace.Trace}
    @type dt_resample: float
    @rtype: L{obspy.core.trace.Trace}
    """
    dt = 1.0 / trace.stats.sampling_rate
    factor = dt_resample / dt
    if int(factor) == factor:
        # simple decimation (no filt because it shifts the data)
        trace.decimate(int(factor), no_filter=True)
    else:
        # linear interpolation
        tp = np.arange(0, trace.stats.npts) * trace.stats.delta
        zp = trace.data
        ninterp = int(max(tp) / dt_resample) + 1
        tinterp = np.arange(0, ninterp) * dt_resample

        trace.data = np.interp(tinterp, tp, zp)
        trace.stats.npts = ninterp
        trace.stats.delta = dt_resample
        trace.stats.sampling_rate = 1.0 / dt_resample
        # trace.stats.endtime = trace.stats.endtime + max(tinterp)-max(tp)


def auto_wind(
    rtr, ztr, dt, freqrange=(0.008, 0.012), maxdeg=20, previous_bound=None, debug=False
):
    # Find the left bound
    minfreq, maxfreq = freqrange
    numT = int(2 / (minfreq + maxfreq) / dt)

    # on the left: a half cycle is also assumed to be the signal and taper
    # before the windowed signal
    # filter the seismogram
    rbp, zbp = deepcopy(rtr), deepcopy(ztr)
    rbp.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)
    zbp.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)

    # find the logest segments with the stable polarization
    # obtain analytic function of signals
    analytic_rbp = scipy.signal.hilbert(rbp.data)
    analytic_zbp = scipy.signal.hilbert(zbp.data)


    # phase angle
    RZ_phase_differences = np.rad2deg(np.angle(analytic_rbp / analytic_zbp))
    counts, nonzeroidx = check_longest_stable_period(
        RZ_phase_differences, mean=90, maxstd=maxdeg, debug=debug
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(rbp.data, label="R")
    axes[0].plot(zbp.data, label="Z")
    axes[1].plot(RZ_phase_differences, label="diff")
    axes[0].legend()
    axes[1].legend()
    plt.show()
    plt.close()
    
    idx = counts.argmax()
    minidx = nonzeroidx[idx]
    maxidx = minidx + counts[idx]

    # Find segments with the maximum
    if previous_bound:
        count_intervals = np.array(
            [
                Interval(nonzeroidx[idx], nonzeroidx[idx] + counts[idx])
                for idx in range(len(counts))
            ]
        )
        count_intercepts = np.array([x & previous_bound for x in count_intervals])
        overlap_length = np.array(
            [x.upper_bound - x.lower_bound for x in count_intercepts]
        )
        seg_idx = overlap_length.argmax()

        # Assign the segment with maximum overlap region to be signal period of
        # the current frequency range
        minidx = nonzeroidx[seg_idx]
        maxidx = minidx + counts[seg_idx]

    # setup the window parameters
    if (maxidx - minidx) <= numT:
        return None
    else:
        lft_bound = int(minidx - 0.5 * numT)
        rgt_bound = int(maxidx + 0.5 * numT)
    return Interval(lft_bound, rgt_bound)


@jit(nopython=True)
def time_window(
    times, lft_bound, lft_corner, lft_radius, rgt_bound, rgt_corner, rgt_radius
):
    """define the welch window used for constructure filter
    """
    # Compute decay function based on the welch window
    responses = np.zeros_like(times)
    for idx, time in enumerate(times):
        if (time >= lft_bound) and (time <= lft_corner):
            responses[idx] = 1 - ((time - lft_corner) / lft_radius) ** 2
        elif (time >= lft_corner) and (time <= rgt_corner):
            responses[idx] = 1
        elif (time >= rgt_corner) and (time <= rgt_bound):
            responses[idx] = 1 - ((time - rgt_corner) / rgt_radius) ** 2
    return responses

@jit(nopython=True)
def freq_window(
    freqs, lft_bound, lft_corner, lft_radius, rgt_bound, rgt_corner, rgt_radius
):
    """define the welch window used for constructure filter
    """
    # Compute decay function based on the welch window
    responses = np.ones_like(freqs) * 10**(-9)
    for idx, freq in enumerate(freqs):
        if (freq >= lft_bound) and (freq <= lft_corner):
            responses[idx] = 1 - ((freq - lft_corner) / lft_radius) ** 2
        elif (freq >= lft_corner) and (freq <= rgt_corner):
            responses[idx] = 1
        elif (freq >= rgt_corner) and (freq <= rgt_bound):
            responses[idx] = 1 - ((freq - rgt_corner) / rgt_radius) ** 2
    return responses

@jit(nopython=True)
def phase_shift(iptsignal, angle, dt):
    """Perform phase shift of arbitary angle

    Parameter
    =========
    iptsignal : numpy.array
        input signal
    angle : float
        angle to shift signal, in degree
    dt : float
        time step
    """
    # obtain spectrum
    spec = np.fft.rfft(iptsignal)

    # ampl, angle = np.abs(spec), np.angle(spec)
    freq = np.fft.rfftfreq(iptsignal.size, d=dt)
    spec[freq > 0] *= np.exp(1j * np.deg2rad(angle))
    spec[freq < 0] *= np.exp(-1.0j * np.deg2rad(angle))
    phaseshift = np.fft.irfft(spec, n=len(iptsignal))
    return phaseshift


@jit(nopython=True)
def check_longest_stable_period(phase_shift_array, mean, maxstd=10, debug=True):
    """check the longest time where the phase shift is stable around 
    the 90 degree

    Parameter
    =========
    phase_shift_array : numpy.array
        array contains phase shifts of each times
    maxstd : float
        maximum stantard deviation from 90 degree
    mean : float
        mean value to be searched 
    """
    # search for the maximum stable length
    length = len(phase_shift_array)
    counts = np.array([0 for _ in range(length)])

    nonzeroidx = []
    maskarray = np.abs(phase_shift_array - mean) < maxstd
    for i in range(0, length):
        # count stable data number after ith point
        for j in range(i, length):
            if maskarray[j]:
                counts[i] += 1
                nonzeroidx.append(i)
            else:
                # continous break
                break
    msk = counts != 0
    return counts[msk], np.array(nonzeroidx)

def signal_ratio_measurement(obs, origin, refDispFunc, freqmin, freqmax, fwidth=0.01):
    """Isolate the Rayleigh wave or Love wave with the reference phase velocity 
    dispersion curve

    Parameters
    ==========
    refDisp: tuple
        reference dispersion curve
    freqmin: float
        minimum frequency
    freqmax: float
        maximum frequency
    """
    # Filtering the seismograms
    obs.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

    # Construct phase matched filter
    #reff, refc = refDisp
    #fck = interp1d(reff, refc, kind="linear", fill_value="extrapolate")

    dataNum, dt = len(obs.data), 1.0/obs.stats.sampling_rate
    obsfreq = rfftfreq(dataNum, d=dt)
    msk = (obsfreq >= freqmin) * (obsfreq <= freqmax)
    obsrefc = refDispFunc(obsfreq[msk])

    stats = obs.stats
    common_T = (stats.endtime - stats.starttime) / 2
    common_T+= (stats.starttime - origin)
    psip = 2 * np.pi * obsfreq[msk] * stats.sac.dist / obsrefc
    psic = 2 * np.pi * obsfreq[msk] * common_T
    shiftphase = np.exp((psip - psic) * 1j)

    # Shift the spectrum phase
    obsspec = rfft(obs.data)
    obsspec[msk] *= shiftphase

    # Back to time domain
    window = windowFunc(width=int(4.0 / (freqmin * dt)), n=dataNum)
    shiftobs = irfft(obsspec, n=dataNum) * window

    shiftobsTr = deepcopy(obs)
    shiftobsTr.data = shiftobs

    # Obtain the spectum
    spectrum = np.abs(rfft(shiftobs))
    obsfreq = rfftfreq(dataNum, d=dt)

    # Compute signal ratio
    ratios = signalRatio(obsfreq, spectrum, np.abs(obsspec), fwidth, freqmin, freqmax)
    fx = interp1d(obsfreq, ratios, kind="linear", fill_value="extrapolate")
    return fx 

def signalRatio(freqs, shiftspec, obsspec, fwidth, fmin, fmax):
    """Compute the signal ratio

    Parameters
    ==========
    freqs: numpy.array
        frequency
    shiftspec: numpy.array
        spectrum of the shifted trace
    obsspec: numpy.array
        spectrum of the observed trace
    fwidth: float
        smooth width of each frequency
    fmin: float
        minimum frequency
    fmax: float
        maximum frequency
    """
    ratios = np.ones(len(freqs)) * np.nan
    for idx, freq in enumerate(freqs):
        if freq < fmin or freq > fmax:
            continue
        minfreq, maxfreq = freq-fwidth, freq+fwidth
        msk = (freqs >= minfreq) * (freqs <= maxfreq)
        ratios[idx] = shiftspec[msk].mean() / obsspec[msk].mean()
    return ratios

def windowFunc(width, n):
    """Construct the time-domain filter function to isolate the fundamental-mode
    surface wave
    """
    responses = np.zeros(n)
    window = scipy.signal.windows.hann(width)
    center = n // 2
    
    start = center-width//2
    end = start + width
    responses[start: end] = window
    return responses

if __name__ == "__main__":
    xax = np.arange(0, 10, 0.5)
    v = np.arange(0, 10, 0.5) / 10 + 3
    coeff = fit_value_with_spline_basis(xax, v)["x"]

    # Visualize
    y = dmatrix("bs(x, df=4, degree=3, include_intercept=True) - 1", {"x": xax})
    rec_v = np.dot(y, coeff)

    plt.plot(xax, y * coeff)
    plt.plot(xax, rec_v, color="k", lw=3, label="recovered")
    plt.plot(xax, v, "o", label="Raw")
    plt.xlabel("Thickness (km)")
    plt.ylabel("Vs (km/s)")
    plt.legend()
    plt.show()
    plt.close()

    dt = 0.2
    t = np.arange(0, 10000, dt)
    pi2 = np.pi * 2
    a = np.cos(pi2 * t)
    freq = np.fft.fftfreq(len(a), d=dt)
    plt.plot(freq, np.fft.fft(a), "o")
    plt.plot(1, DFT(a, dt=dt, f=1), "o")
    plt.show()
