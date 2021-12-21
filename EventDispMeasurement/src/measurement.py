# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 16:36h, 20/08/2020
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2020-2021 Xiao Xiao
# -------------------------------------------------------------------------------
"""Modulues used for measuring dispersion curve of a specific seismic station
"""
from . import logger
from .utils import (
    fit_value_with_spline_basis, 
    welch_window, 
    resample, 
    freq_window, 
    signal_ratio_measurement)

from obspy import read
from obspy.io.sac import SACTrace
from scipy.optimize import minimize
from numpy.fft import rfft, rfftfreq, irfft, fftshift
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from functools import partial
import os
import tempfile
from os.path import join
from easyprocess import EasyProcess
import shutil
import subprocess
from copy import deepcopy

plt.switch_backend("agg")


class measurement(object):
    """class including information and measurements of a specific seismic station
    during an event 
    """

    def __init__(self, eventid, trdir=None, wavetype="Rayleigh"):
        """initiation of the event with a ID of eventid

        Parameters
        ==========
        eventid: str
            ID of the event, YYYYMMDDHHMMSS
        trdir: str.
            directories of a seismic trace
        """
        self.eventid = eventid
        self.wavetype = wavetype
        if self.wavetype == "Rayleigh":
            self.comp = "Z"
        elif self.wavetype == "Love":
            self.comp = "T"

        if trdir:
            self.load_seismogram(trdir=trdir)

    def load_seismogram(
        self,
        trdir,
        downsample=False,
        m2cm=False,
        nm2cm=False,
        dt=None,
        minfreq=0.008,
        maxfreq=1,
        mindist=None,
        maxdist=None,
        removeNan=True,
    ):
        """load the seismograms of during the specific event recorded by multiple
        seismic stations

        Parameters
        ==========
        trdir: str.
            directories of the seismic trace
        """
        logger.info("Loading {}".format(trdir))
        try:
            tr = read(trdir)[0]
            dist = tr.stats.sac["dist"]
        except ValueError as err:
            logger.error("Unhandled error [{}]".format(err))

        # filter with epicentral distance
        if mindist:
            if dist <= mindist:
                fmt = "trace epicentral distance too small {} during {}"
                raise ValueError(fmt.format(trdir, self.eventid))
        if maxdist:
            if dist >= maxdist:
                fmt = "trace epicentral distance too large {} during {}"
                raise ValueError(fmt.format(trdir, self.eventid))


        # store the epicentral distance
        self.dist = dist
        if downsample:
            tr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)
            resample(tr, dt_resample=dt)

        # unit transfer
        if m2cm:
            tr.data *= 100

        if nm2cm:
            tr.data /= 10 ** 7

        if removeNan:
            # Use the interpolation to fill the result
            msk = np.isnan(tr.data)
            if msk.any():
                times, data = tr.times(), tr.data
                fx = interp1d(times[~msk], data[~msk])
                tr.data[msk] = fx(times[msk])

        # Obtain the origin
        demotr = SACTrace.from_obspy_trace(tr)
        self.origin, self.dt = demotr.reftime, demotr.delta

        # update the measurements
        key = ".".join(tr.id.split(".")[:3])
        self.obs, self.key = tr, key

    def snr(self, minvg, maxvg, minfreq, maxfreq):
        """Compute signal noise ratio
        """
        # Time axis
        bptr = deepcopy(self.obs)
        bptr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)

        sbtime = self.origin + int(self.dist / maxvg) - bptr.stats.starttime
        setime = self.origin + int(self.dist / minvg) - bptr.stats.starttime

        data = bptr.data
        times = bptr.times()

        # Signal window
        sigmask = (times >= sbtime) * (times <= setime)
        maxSig = np.abs(data[sigmask]).max()
        nosmask = times >= setime
        nosStd = np.std(data[nosmask])
        return maxSig / nosStd

    def syn_surface_wave(self, moddir, source, fmin=0.008, fmax=0.5, verbose=False):
        """Synthetic the rayleigh wave seismogram on the vertical component or 
        the Love wave on the T component
        
        Parameters
        ==========
        moddir: str
            directory of the model
        source: dict.
            source information including the origin, location and moment 
            tensor
        wavetype: str.
            type of surface wave, could be Rayleigh or Love
        fmin: float
            minimal frequency
        fmax: float
            maximal frequency
        dt: float
            delta of time
        """
        if self.wavetype == "Rayleigh":
            waveflag, egnmsg, egnname = "-R", "sregn96 -v", "SREGN.ASC"
        elif self.wavetype == "Love":
            waveflag, egnmsg, egnname = "-L", "slegn96 -v", "SLEGN.ASC"

        # making and moving to temporary dir
        current_dir = os.getcwd()
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)

        shutil.copy2(moddir, "./")
        filename = os.path.basename(moddir)

        # call cps330 to synthetic seismogram
        dp = source["depth"]
        fmt = "sprep96 -M {} {} -NMOD 1 -HS {:.2f} -HR 0 -FMIN {:.6f} -FMAX {:6f} -v"
        p = EasyProcess(fmt.format(filename, waveflag, dp, fmin, fmax)).call()
        if verbose:
            logger.info(p.stdout)

        p = EasyProcess("sdisp96 -v").call()
        if verbose:
            logger.info(p.stdout)

        p = EasyProcess(egnmsg).call()
        if verbose:
            logger.info(p.stdout)

        # Generate the reference dispersion curve
        p = EasyProcess("sdpegn96 {} -E -ASC".format(waveflag)).call()
        if verbose:
            logger.info(p.stdout)

        # loading the reference dispersion curve of the phase velocity c and
        # group velocity u
        freq, c, u, gamma = np.loadtxt(
            egnname, skiprows=1, usecols=(3, 4, 5, 7), unpack=True
        )
        interpf = interp1d(freq, c, kind="linear", fill_value="extrapolate")
        self.refc = {"freq": freq, "c": c, "func": interpf}
        interpf = interp1d(freq, u, kind="linear", fill_value="extrapolate")
        self.refu = {"freq": freq, "u": u, "func": interpf}

        # Load attenuation coefficient
        interpf = interp1d(freq, gamma, kind="linear", fill_value="extrapolate")
        self.gamma = {"freq": freq, "gamma": gamma, "func": interpf}

        Mxx, Myy, Mzz = source["Mxx"], source["Myy"], source["Mzz"]
        Mxy, Mxz, Myz = source["Mxy"], source["Mxz"], source["Myz"]
        demoTr = SACTrace.from_obspy_trace(self.obs)

        # Produce the distance file
        mw, az, delta = source["mag"], demoTr.az, demoTr.delta
        npts = len(demoTr.data)

        msg = "{:.2f} {:.2f} {} 0 8\n".format(self.dist, delta, npts)
        with open("dfile", "w") as f:
            f.write(msg)

        #durationT = math.ceil(source["duration"] / (2 * delta))
        #msg1 = "spulse96 -d dfile -t -V -l {} -v | ".format(durationT)
        msg1 = "spulse96 -d dfile -V -i -v | "
        msg2 = [
            "fmech96",
            "-A {:.2f}".format(az),
            "-ROT",
            "-MW {:.2f}".format(mw),
            "-XX {:.6e}".format(Mxx),
            "-YY {:.6e}".format(Myy),
            "-ZZ {:.6e}".format(Mzz),
            "-XY {:.6e}".format(Mxy),
            "-XZ {:.6e}".format(Mxz),
            "-YZ {:.6e}".format(Myz),
            "-v",
        ]
        msg = msg1 + " ".join(msg2) + " | f96tosac -B"

        p = subprocess.check_output(msg, shell=True)
        if verbose:
            logger.info(p)

        # Obtain vertical/transverse components of the synthetic traces
        synTr = read("*{}*.sac".format(self.comp))[0]

        # Change the origin time
        tmpsactr = SACTrace.from_obspy_trace(synTr)
        oldb = tmpsactr.b
        tmpsactr.reftime = demoTr.reftime
        tmpsactr.b = oldb
        self.syn = tmpsactr.to_obspy_trace()

        # removing temp dir
        os.chdir(current_dir)
        shutil.rmtree(tmp_dir)

    def export_syn(self, exdir):
        """export the synthetic sesimogram in the event ID
        """

        filename = join(exdir, "Syn.{}.{}.SAC".format(self.key, self.comp))
        self.syn.write(filename, format="SAC")

        filename = join(exdir, "Ref.disp.{}.{}.info".format(self.key, self.wavetype))
        exmat = np.matrix(
            [self.refc["freq"], self.refc["c"], self.refu["u"], self.gamma["gamma"]]
        ).T
        np.savetxt(
            filename, exmat, header="Freq(Hz) c(km/s) u(km/s) gamma(1/km)", fmt="%.5e"
        )

    def load_syn(self, dbdir):
        """export the synthetic sesimogram in the event ID
        """
        # load synthetic seismogram
        filename = join(dbdir, "Syn.{}.{}.SAC".format(self.key, self.comp))
        self.syn = read(filename)[0]

        # load synthetic dispersion curves
        filename = join(dbdir, "Ref.disp.{}.{}.info".format(self.key, self.wavetype))
        freq, c, u, gamma = np.loadtxt(filename, unpack=True)
        func = interp1d(freq, c, kind="linear", fill_value="extrapolate")
        self.refc = {"freq": freq, "c": c, "func": func}
        func = interp1d(freq, u, kind="linear", fill_value="extrapolate")
        self.refu = {"freq": freq, "u": u, "func": func}
        func = interp1d(freq, gamma, kind="linear", fill_value="extrapolate")
        self.gamma = {"freq": freq, "gamma": gamma, "func": func}

    def export_fitness(self, exdir, save_plot=True, ex_resample=True):
        """export the measured dispersion curve and its comparison 
        to the reference one
        """
        logger.info("Exporting measurement of {} @ {}".format(self.key, self.eventid))
        fc = self.refc["func"]
        # Export the fitted parameters
        fitted_freq = self.fitted_freq
        fitted_deltac = self.fitted_deltac
        fitted_abs_amp = self.fitted_abs_amp
        fitted_abs_syn_amp = self.fitted_abs_syn_amp
        obs_abs_amp = self.obs_abs_amp
        fitted_refc = fc(fitted_freq)
        fitted_c = fitted_refc + fitted_deltac

        # Export the resampled parameters
        resample_freq = self.resample_freq
        resample_abs_amp = self.resample_abs_amp
        resample_abs_syn_amp = self.resample_abs_syn_amp
        resample_deltac = self.resample_deltac
        resample_refc = fc(resample_freq)
        resample_c = resample_refc + resample_deltac
        amplification = self.amplification
        comp, key = self.comp, self.key
        
        # export the fitness
        obsTr, synTr, ad_Tr = self.obs, self.syn, self.ad_Tr

        # Plot dispetrsion curve
        if save_plot:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 24))
            axes[0].plot(resample_freq, resample_refc, label="Reference Vc")
            axes[0].plot(fitted_freq, fitted_c, "s", label="Fitted Vc")
            axes[0].plot(resample_freq, resample_c, "--", label="Resampled Vc")
            axes[0].set_xlabel("Frequency (Hz)")
            axes[0].set_ylabel("Phase velocity (km/s)")
            axes[0].set_title(self.key)

            axes[1].plot(
                fitted_freq,
                (fitted_c - fitted_refc) * 100 / fitted_refc,
                "s",
                label="Fitted",
            )
            axes[1].plot(
                resample_freq,
                (resample_c - resample_refc) * 100 / resample_refc,
                "--",
                label="Resampled",
            )
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Phase velocity perturbation (%)")

            # Plot the amplitude
            axes[2].plot(resample_freq, resample_abs_amp, label="Resample {}".format(comp))
            axes[2].plot(fitted_freq, fitted_abs_amp, "s", label="Fitted {}".format(comp))
            axes[2].plot(fitted_freq, fitted_abs_syn_amp, label="Synthetic {}".format(comp))
            axes[2].plot(
                fitted_freq,
                fitted_abs_syn_amp * amplification,
                label="Amplified synthetic {}".format(comp),
            )
            axes[2].plot(fitted_freq, obs_abs_amp, label="Obs. {}".format(comp))
            axes[2].set_xlabel("Frequency (Hz)")
            axes[2].set_ylabel("Amplitude")


            # export the fitted seismogram
            filename = join(exdir, "Fitted.{}.{}.waveform.SAC".format(comp, key))
            ad_Tr.write(filename, format="SAC")
            minfreq, maxfreq = self.minfreq, self.maxfreq

            obsTr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)
            synTr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)
            ad_Tr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, zerophase=True)

            origin = SACTrace.from_obspy_trace(synTr).reftime

            timesyn = synTr.times() + (synTr.stats.starttime - origin)
            timeobs = obsTr.times() + (obsTr.stats.starttime - origin)
            timead = ad_Tr.times() + (ad_Tr.stats.starttime - origin)
            mint, maxt = timeobs.min(), timeobs.max()

            axes[3].plot(
                timesyn, synTr.data, "--", color="grey", label="Reference {}".format(comp)
            )
            axes[3].plot(timeobs, obsTr.data, label="Observed {}".format(comp))
            axes[3].plot(timead, ad_Tr.data, label="Fitted {}".format(comp))
            axes[3].set_title(key)
            axes[3].set_xlabel("Time from origin(s)")
            axes[3].set_ylabel("Amplitude(cm/s)")
            axes[3].set_xlim((mint, maxt))

            for idx in range(4):
                axes[idx].legend()

            exfilename = join(exdir, "Fitness.{}.{}.pdf".format(comp, key))
            fig.savefig(exfilename)
            plt.close()

        # Export the measured dispersion curve
        exmat = np.matrix(
            [
                1.0 / fitted_freq[::-1],
                fitted_c[::-1],
                fitted_abs_amp[::-1],
                fitted_abs_syn_amp[::-1],
                obs_abs_amp[::-1],
                self.snrs[::-1],
            ]
        ).T
        filename = join(exdir, "Fitted.param.{}.{}.info".format(comp, key))

        header = [
            "Amplification factor: {:.5f}".format(amplification),
            "Period (s) Fitted_Vc(km/s) Fitted_amp Synthetic_amp Obs_amp snr",
        ]
        header = "\n".join(header)
        np.savetxt(filename, exmat, fmt="%.5f", header=header)

        # Export the measured dispersion curve
        if ex_resample:
            exmat = np.matrix(
                [
                    1.0 / resample_freq[::-1],
                    resample_c[::-1],
                    resample_abs_amp[::-1],
                    resample_abs_syn_amp[::-1],
                ]
            ).T
            filename = join(exdir, "Resample.param.{}.{}.info".format(comp, key))
            np.savetxt(
                filename,
                exmat,
                fmt="%.5f",
                header="Period (s) Resample_Vc(km/s) Resample_amp Synthetic_amp",
            )

        # save the inversion message
        x = self.misfit
        exmsg = [
            "success: {}".format(x["success"]),
            "nit: {}".format(x["nit"]),
            "misfit: {}".format(x["fun"]),
            "deltac variation: {:.5f} # Percentage".format(self.dc),
            "frequency band: {:.5f} {:.5f} # Hz".format(self.minfreq, self.maxfreq),
            "group velocity range: {:.5f} {:.5f} # km/s".format(self.minvg, self.maxvg),
            "dist: {:.5f} # Km".format(obsTr.stats.sac["dist"]),
            "baz: {:.5f} # deg".format(obsTr.stats.sac["baz"]),
            "minlen, maxlen: ({:.5f}, {:.5f}) # s".format(self.minlen, self.maxlen),
            "amplification_mag: {:.5f} # Percentage".format(self.amplification_mag),
            "amp_variation: {:.5f} # Percentage".format(self.amp_variation),
            "freq_lower(Hz) freq_upper(Hz) sub_misfit",
        ]

        frequency_bands = self.frequency_bands
        sub_misfits = self.sub_misfits
        for idx, x in enumerate(frequency_bands):
            submsg = "{:.5f} {:.5f} {:.5f}".format(
                x[0], x[1], sub_misfits[idx]
            )
            exmsg.append(submsg)

        filename = join(exdir, "Fiteness.{}.{}.message.info".format(comp, key))
        with open(filename, "w") as f:
            f.write("\n".join(exmsg))

    def measure_dispersion_curve(
        self,
        minfreq=0.01,
        maxfreq=0.04,
        maxlen=1000,
        minlen=400,
        df=0.001,
        N=3,
        dc=0.1,
        amplification_mag=10,
        amp_variation=0.5,
        maxvg=5.5,
        minvg=1,
        velo2disp=False,
        verbose=False,
    ):
        """measure diposersion curve of the observed seismic trace based on the 
        given reference
        """
        logger.info(
            "Measuring {} of {} during {}".format(self.wavetype, self.key, self.eventid)
        )

        # Attach measurement configuration
        self.minfreq, self.maxfreq = minfreq, maxfreq
        self.df, self.maxvg, self.minvg = df, maxvg, minvg
        self.dc, self.amplification_mag = dc, amplification_mag
        self.amp_variation, self.maxlen, self.minlen = amp_variation, maxlen, minlen
        synTr, obsTr = self.syn, self.obs

        if velo2disp:
            obsTr.integrate()
            synTr.integrate()

        dist, dt = self.dist, self.dt

        # Preprocess the observed and synthetic seismograms
        for tr in [obsTr, synTr]:
            tr.detrend("constant")
            tr.detrend("linear")

        # window the observation
        if maxvg:
            origin = self.origin
            starttime, endtime = origin + int(dist / maxvg), origin + int(dist / minvg)
            if (endtime - starttime) >= maxlen:
                endtime = starttime + maxlen
            elif (endtime - starttime) <= minlen:
                endtime = starttime + minlen

            for tr in [obsTr, synTr]:
                tr.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0.0)

            if len(obsTr.data) != len(synTr.data):
                starttime = max((obsTr.stats.starttime, synTr.stats.starttime))
                endtime = min((obsTr.stats.endtime, synTr.stats.endtime))
                for tr in [obsTr, synTr]:
                    tr.trim(
                        starttime=starttime, endtime=endtime, pad=True, fill_value=0.0
                    )


        dataNum = len(obsTr.data)
        synfreq = rfftfreq(dataNum, d=dt)

        # Length of the frequency band
        dw = (maxfreq - minfreq) / (2 * N)

        # Fit the observed spectral amplitude
        freq_bands = [
            (minfreq + (idx - 1) * dw, minfreq + (idx + 1) * dw)
            for idx in range(2 * N + 1)
        ]
        amp_filter = freq_window(
            synfreq, minfreq - dw, minfreq, dw, maxfreq + 2 * dw, maxfreq + dw, dw
        )

        # Compute spectrum of the observation
        obs_spec = rfft(obsTr.data) * amp_filter
        obs_spec_amp = np.abs(obs_spec)

        # obtain the synthetic information
        syn_spec = rfft(synTr.data) * amp_filter
        syn_spec_psi = np.angle(syn_spec)
        syn_spec_amp = np.abs(syn_spec)

        # interpolate the phase velocity
        c0 = self.refc["func"](synfreq)

        # Find the maximum in the observed radial and vertical components to
        # account for systematical amplification factor from the synthetic to
        # to the data
        mask = (synfreq >= (minfreq - dw)) * (synfreq <= (maxfreq + dw))

        # Linear regression to find the amplification factot
        func = lambda x, a: a * x
        bounds, p0 = (1.0 / amplification_mag, amplification_mag), 1
        popt, _ = curve_fit(
            func, syn_spec_amp[mask], obs_spec_amp[mask], bounds=bounds, p0=p0
        )
        amplification = popt[0]

        # Construct filter in different frequency band
        filter_funcs = [welch_window(synfreq, x[0], x[1]) for x in freq_bands]
        tHws = [fftshift(irfft(x)) for x in filter_funcs]

        # Optimize to find the best spline function coefficients to fit the observed
        # seismogram
        for idx in range(2 * N + 1):
            # Setup the frequency for inversion
            sub_freq_bands = freq_bands[0 : idx + 2]
            sub_filter_funcs = filter_funcs[0 : idx + 2]

            lft_bound = sub_freq_bands[0][0]
            right_bound = sub_freq_bands[-1][1]
            mask = (synfreq >= lft_bound) * (synfreq <= right_bound)
            maskedfreq = synfreq[mask]
            splineBasis = dmatrix(
                "bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": maskedfreq}
            )

            # Calculate the dispersion curve boundaries
            dc_mag = c0[mask].max() * dc
            min_dc, max_dc = -1.0 * dc_mag, dc_mag
            dc_ranges = [
                (min_dc, max_dc),
                (min_dc, max_dc),
                (min_dc, max_dc),
                (min_dc, max_dc),
                (min_dc, max_dc),
                (min_dc, max_dc),
            ]

            amp_mag = obs_spec_amp[mask].max()
            amp_ranges = [
                (-1 * amp_mag, amp_mag),
                (-1 * amp_mag, amp_mag),
                (-1 * amp_mag, amp_mag),
                (-1 * amp_mag, amp_mag),
                (-1 * amp_mag, amp_mag),
                (-1 * amp_mag, amp_mag),
            ]

            if idx == 0:
                guess_deltac_coef = np.array([0, 0, 0, 0, 0, 0])
                # guess_amp_coef = np.array([0, 0, 0, 0, 0, 0])
                guess_amp = obs_spec_amp[mask] - syn_spec_amp[mask] * amplification
                guess_amp_coef = fit_value_with_spline_basis(
                    maskedfreq, guess_amp, maskedfreq
                )["x"]
            else:
                # extrapolate the deltac
                guess_deltac_coef = expand_data(
                    maskedfreq, fitted_freq, dw, fitted_deltac
                )

                # expand the amplitude
                delta_amp = obs_spec_amp[mask] - syn_spec_amp[mask] * amplification
                extramsk = maskedfreq > fitted_freq[-1]
                guess_amp = np.zeros_like(maskedfreq)
                guess_amp[~extramsk] = fitted_amp
                guess_amp[extramsk] = delta_amp[extramsk]
                guess_amp_coef = fit_value_with_spline_basis(
                    maskedfreq, guess_amp, maskedfreq
                )["x"]

                # guess_amp_coef = expand_data(
                #    maskedfreq, fitted_freq, dw, fitted_amp
                # )

            x0 = np.array(list(guess_amp_coef) + list(guess_deltac_coef))

            part_misfit = partial(
                compute_misfit,
                synfreq=synfreq,
                mask=mask,
                c0=c0,
                splineBasis=splineBasis,
                sub_filter_funcs=sub_filter_funcs,
                tHws=tHws,
                dataNum=dataNum,
                dist=dist,
                dt=dt,
                syn_spec_amp=syn_spec_amp,
                amplification=amplification,
                syn_spec_psi=syn_spec_psi,
                obs_spec=obs_spec,
                sub_freq_bands=sub_freq_bands,
            )
            bnds = amp_ranges + dc_ranges
            options = {"maxiter": 1000, "ftol": 1e-8}
            x = minimize(part_misfit, x0, method="SLSQP", bounds=bnds, options=options)

            # Extract the fitted amplitude and phase velocity perturbation
            fitted_amp_coef = x["x"][0:6]
            fitted_deltac_coef = x["x"][6:]

            fitted_deltac = np.dot(splineBasis, fitted_deltac_coef)
            fitted_amp = np.dot(splineBasis, fitted_amp_coef)
            fitted_freq = maskedfreq

        # Compute misfit, snr and seismograms of the trial model
        sub_misfits, Tr_data = sub_synthetic(
            mods=x["x"],
            dataNum=dataNum,
            synfreq=synfreq,
            mask=mask,
            splineBasis=splineBasis,
            sub_filter_funcs=sub_filter_funcs,
            tHws=tHws,
            c0=c0,
            dist=dist,
            dt=dt,
            syn_spec_amp=syn_spec_amp,
            amplification=amplification,
            syn_spec_psi=syn_spec_psi,
            obs_spec=obs_spec,
            sub_freq_bands=sub_freq_bands,
            conclude_computation=True,
        )
        ad_Tr = deepcopy(obsTr)
        ad_Tr.data = Tr_data

        # Compute signal portion as signal noise ratio
        ratiofunc = signal_ratio_measurement(
            obs=deepcopy(obsTr), 
            origin=origin,
            refDispFunc=self.refc["func"], 
            freqmin=fitted_freq[0], 
            freqmax=fitted_freq[-1], 
            fwidth=0.01
        )

        # filter the adjusted seismogram
        ad_Tr.filter(
            "bandpass", freqmin=fitted_freq[0], freqmax=fitted_freq[-1], zerophase=True
        )
        self.ad_Tr, self.misfit, self.frequency_bands = ad_Tr, x, freq_bands
        self.sub_misfits = sub_misfits

        # resample the fitted parameters
        resample_freq = np.arange(fitted_freq[0], fitted_freq[-1] + df, df)
        splineBasis = dmatrix(
            "bs(x, df=6, degree=3, include_intercept=True) - 1", {"x": resample_freq}
        )

        # Mask ranges
        left_freq, right_freq = fitted_freq[0], fitted_freq[-1]
        msk1 = (fitted_freq >= left_freq) * (fitted_freq <= right_freq)
        msk2 = (synfreq >= left_freq) * (synfreq <= right_freq)
        msk3 = (resample_freq >= left_freq) * (resample_freq <= right_freq)

        # Resample the fitted results
        resample_amp = np.dot(splineBasis, fitted_amp_coef)
        resample_deltac = np.dot(splineBasis, fitted_deltac_coef)

        # Interpolate to obtain the absolute resampled amplitude on the RZ component
        fx = interp1d(
            synfreq[msk2], syn_spec_amp[msk2], kind="linear", fill_value="extrapolate"
        )
        resample_abs_syn_amp = fx(resample_freq)
        fx = interp1d(
            synfreq[msk2], syn_spec_amp[msk2], kind="linear", fill_value="extrapolate"
        )
        resample_abs_syn_amp = fx(resample_freq)

        toarr = lambda x: np.array([z for z in x])
        # Obtain the resampled amplitudes
        resample_abs_amp = toarr(resample_abs_syn_amp * amplification + resample_amp)
        self.resample_freq = toarr(resample_freq[msk3])
        self.resample_deltac = toarr(resample_deltac[msk3])
        self.resample_abs_syn_amp = toarr(resample_abs_syn_amp[msk3])
        self.resample_abs_amp = toarr(resample_abs_amp[msk3])
        msk = self.resample_abs_amp <= 0
        self.resample_abs_amp[msk] = 10 ** (-10)
        self.amplification = amplification

        # THe raw sampling rate in frequency domain determined by data
        self.fitted_abs_amp = toarr(
            (syn_spec_amp[mask] * amplification + fitted_amp)[msk1]
        )
        msk = self.fitted_abs_amp <= 0
        self.fitted_abs_amp[msk] = 10 ** (-10)
        self.fitted_abs_syn_amp = toarr(syn_spec_amp[mask][msk1])

        self.obs_abs_amp = toarr(obs_spec_amp[mask][msk1])
        self.fitted_freq = toarr(fitted_freq[msk1])
        self.fitted_deltac = toarr(fitted_deltac[msk1])
        self.snrs = ratiofunc(self.fitted_freq)


def compute_misfit(
    mods,
    dataNum,
    synfreq,
    mask,
    splineBasis,
    sub_filter_funcs,
    tHws,
    c0,
    dist,
    dt,
    syn_spec_amp,
    amplification,
    syn_spec_psi,
    obs_spec,
    sub_freq_bands,
):
    """Compute misfit of a given perturbed phase velocity dispersion curve
    """
    sub_misfits = sub_synthetic(
        mods=mods,
        dataNum=dataNum,
        synfreq=synfreq,
        mask=mask,
        splineBasis=splineBasis,
        sub_filter_funcs=sub_filter_funcs,
        tHws=tHws,
        c0=c0,
        dist=dist,
        dt=dt,
        syn_spec_amp=syn_spec_amp,
        amplification=amplification,
        syn_spec_psi=syn_spec_psi,
        obs_spec=obs_spec,
        sub_freq_bands=sub_freq_bands,
    )

    # temporarily set the weights to be an unity
    wts = np.ones_like(sub_misfits) / len(sub_misfits)
    misfit = (sub_misfits * wts).sum()
    return misfit


def sub_synthetic(
    mods,
    dataNum,
    synfreq,
    mask,
    splineBasis,
    sub_filter_funcs,
    tHws,
    c0,
    dist,
    dt,
    syn_spec_amp,
    amplification,
    syn_spec_psi,
    obs_spec,
    sub_freq_bands,
    conclude_computation=False,
):
    """Compute trial model and associated uncertainty of a given perturbed phase 
    velocity dispersion curve
    """
    # Construct the trial model
    mod_amp_coef = mods[0:6]
    mod_deltac_coef = mods[6:]

    # Construct synthetic spectrum with another set of coefficients
    maskedfreq = synfreq[mask]

    # Compute difference of propagation phase, induced by the variation
    # of the apparent phase velocity
    mod_deltac = np.dot(splineBasis, mod_deltac_coef)
    delta_psi = 2 * np.pi * maskedfreq * dist
    delta_psi *= 1 / (c0[mask] + mod_deltac) - 1 / c0[mask]

    # Synthetic the trial seismogram
    # In fourier transform, the kernel is e^(-i omega t), thus we need to add
    # negtive of the phase perturbation, please refer to:
    # https//blog.seispider.top/post/2018-07-29-phase-shift/
    mod_psi = deepcopy(syn_spec_psi)
    mod_psi[mask] += -1.0 * delta_psi
    mod_phase = np.array([np.exp(x * 1j) for x in mod_psi])

    # Perturb the amplitude with residual attenuation coefficients
    mod_amp_mask = np.dot(splineBasis, mod_amp_coef)
    mod_amp = deepcopy(syn_spec_amp) * amplification
    mod_amp[mask] += mod_amp_mask

    msk2 = mod_amp[mask] < 0
    mod_amp[mask][msk2] = 10 ** (-10)
    # if len(mod_amp[mask][msk2]) != 0:
    #    submisfits = [np.random.rand() * 10000]
    #    print(submisfits)
    #    return submisfits

    # Construct matched filter
    W = (1.0 / mod_amp) * mod_phase

    # obtain cross-correlation and autocorrelation functions
    Gw = obs_spec * np.conjugate(W)

    # obtain misfit function in a single frequency range
    sub_misfits = []
    cidx = dataNum // 2
    for idx, sub_freq_band in enumerate(sub_freq_bands):

        # Filter the data
        left_f, right_f = sub_freq_band
        filter_func = sub_filter_funcs[idx]
        tGw = fftshift(irfft(Gw * filter_func))

        # Obtain the band-passed delta function
        tHw = tHws[idx]

        # define the time range used for construting misfit function
        center_T_nsample = int(2 / ((left_f + right_f) * dt))
        Jnum = int(1.5 * center_T_nsample)

        # Compute the misfit range
        minidx, maxidx = cidx - Jnum, cidx + Jnum
        subtGw = tGw[minidx:maxidx]
        subtHw = tHw[minidx:maxidx]

        # Compare data and trial model
        sub_misfit = ((subtGw - subtHw) ** 2).sum() / ((subtGw) ** 2).sum()
        sub_misfits.append(sub_misfit)

    sub_misfits = np.array(sub_misfits)
    if conclude_computation:
        # Compute the seismogram of trial model
        ad_spec = mod_amp * mod_phase
        Tr_data = irfft(ad_spec, n=dataNum)
        return sub_misfits, Tr_data
    else:
        return sub_misfits


def expand_data(maskedfreq, fitted_freq, dw, fitted_value, fill_value=None):
    # extrapolate the deltac
    maxfreq = fitted_freq[-1]
    extramsk = maskedfreq > maxfreq

    # Maintain values in the fitted frequency range
    guess_value = np.zeros_like(maskedfreq)
    guess_value[~extramsk] = fitted_value

    if fill_value:
        guess_value[extramsk] = fill_value
    else:
        # Use the valid area
        valid_minf = fitted_freq[0] + 0.5 * dw
        valid_maxf = fitted_freq[-1] - dw
        valid_msk = (fitted_freq >= valid_minf) * (fitted_freq <= valid_maxf)
        valid_fitted_freq = fitted_freq[valid_msk]

        # Assign values in the neighbor fitted frequency range
        if len(valid_fitted_freq) == 0:
            neighbor_mean = 0.0
        else:
            neighbor_msk = valid_fitted_freq >= valid_fitted_freq[-1] - 2 * dw
            neighbor_mean = fitted_value[valid_msk][neighbor_msk].mean()
        guess_value[extramsk] = neighbor_mean
    guess_value_coef = fit_value_with_spline_basis(maskedfreq, guess_value, maskedfreq)[
        "x"
    ]
    return guess_value_coef
