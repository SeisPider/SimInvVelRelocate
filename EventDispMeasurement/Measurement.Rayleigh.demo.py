# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 19:52h, 21/08/2020
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2020-2021 Xiao Xiao
# -------------------------------------------------------------------------------
import sys

import numpy as np
from src.measurement import measurement
import multiprocessing as mp
from os.path import join, exists, basename
from obspy import UTCDateTime
from functools import partial
import os
from glob import glob
from src import logger


def load_sources(filename):
    """load seismic source moment tensor and so on.
    """
    with open(filename) as f:
        lines = f.readlines()

    sources = []
    for line in lines:
        parts = line.strip().split()
        origin = UTCDateTime(parts[0])
        lat, lon = np.float(parts[1]), np.float(parts[2])
        depth, mag = np.float(parts[3]), np.float(parts[4])
        Mxx, Myy, Mzz = np.float(parts[6]), np.float(parts[7]), np.float(parts[8])
        Mxy, Mxz, Myz = np.float(parts[9]), np.float(parts[10]), np.float(parts[11])
        duration = np.float(parts[12])
        subsrc = {
            "origin": origin,
            "eventid": origin.strftime("%Y%m%d%H%M%S"),
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "mag": mag,
            "Mxx": Mxx,
            "Myy": Myy,
            "Mzz": Mzz,
            "Mxy": Mxy,
            "Mxz": Mxz,
            "Myz": Myz,
            "duration": duration,
        }
        sources.append(subsrc)
    return sources


def staevent(
    staid,
    source,
    area=(93, 120, 10, 50),
    minsnr=3,
    minfreq=0.018,
    maxfreq=0.06,
    minvg=1.5,
    maxvg=4.5,
    maxlen=2000,
    minlen=1000,
    N=4,
    dc=0.2,
    df=0.0001,
    amplification_mag=10,
    amp_variation=0.99,
    velo2disp=False,
    verbose=True,
    m2cm=True,
):
    try:
        eventid = source["eventid"]

        # Check the existence
        # if (
        #     len(
        #         glob(
        #             "/Users/xiaoxiao/Git/SimInvVelRelocate/EventDispMeasurement/measured/{}/Rayleigh/Fiteness.Z*{}*.message.info".format(
        #                 eventid, staid
        #             )
        #         )
        #     )
        #     != 0
        # ):
        #     logger.info("Ignore {} in {}".format(staid, eventid))
            #return

        datadir = "/Users/xiaoxiao/Git/SimInvVelRelocate/EventDispMeasurement/data"
        trdir = join(datadir, eventid, "{}*?HZ.SAC".format(staid))

        mes = measurement(eventid=eventid, wavetype="Rayleigh")
        mes.load_seismogram(
            trdir=trdir,
            m2cm=m2cm,
            mindist=0,
            maxdist=4000,
            downsample=True,
            dt=1,
            minfreq=0.008,
            maxfreq=0.4,
        )
        minlon, maxlon, minlat, maxlat = area
        stlo, stla = mes.obs.stats.sac["stlo"], mes.obs.stats.sac["stla"]
        if stlo >= maxlon or stlo <= minlon:
            logger.info("Outside area ({},{},{},{})".format(minlon, maxlon, minlat, maxlat))
            return
        if stla >= maxlat or stla <= minlat:
            logger.info("Outside area ({},{},{},{})".format(minlon, maxlon, minlat, maxlat))
            return


        # Filter seismograms with signal noise ratio
        snr = mes.snr(minvg=minvg, maxvg=maxvg, minfreq=minfreq, maxfreq=maxfreq)
        if snr <= minsnr:
            logger.info("Low SNR data [ignore]")
            return

        exdir = "/Users/xiaoxiao/Git/SimInvVelRelocate/EventDispMeasurement/measured/{}/Rayleigh".format(
            eventid
        )
        os.makedirs(exdir, exist_ok=True)

        resynthetic = False
        try:
            mes.load_syn(dbdir=exdir)
            if mes.obs.stats.sampling_rate != mes.syn.stats.sampling_rate:
                logger.info("Different sampling rates")
                resynthetic = True
        except FileNotFoundError as err:
            resynthetic = True

        if resynthetic:
            sachd = mes.obs.stats.sac
            event_locID = "{}_{}".format(int(sachd["evlo"]), int(sachd["evla"]))
            station_locID = "{}_{}".format(int(sachd["stlo"]), int(sachd["stla"]))

            moddir = "/Users/xiaoxiao/Git/SimInvVelRelocate/EventDispMeasurement/info/refmodel"
            filename = join(moddir, event_locID, "{}_cps.mod".format(station_locID))
            mes.syn_surface_wave(filename, source, fmin=0.002, fmax=0.4, verbose=False)
            mes.export_syn(exdir=exdir)

        # export the synthetic seismograms
        mes.measure_dispersion_curve(
            minfreq=minfreq,
            maxfreq=maxfreq,
            minvg=minvg,
            maxvg=maxvg,
            maxlen=maxlen,
            minlen=minlen,
            N=N,
            dc=dc,
            df=df,
            amplification_mag=amplification_mag,
            amp_variation=amp_variation,
            velo2disp=velo2disp,
            verbose=verbose,
        )
        mes.export_fitness(exdir=exdir, save_plot=True, ex_resample=False)
    except Exception as err:
        logger.error("Unhandled Error ({})".format(err))


if __name__ == "__main__":
    filename = "./info/test.info"
    sources = load_sources(filename)

    for source in sources:
        eventid = source["eventid"]
        datadir = "./data/{}".format(eventid)

        # Check the seismic stations
        stafiles = glob(join(datadir, "CB.WHN*HZ.SAC"))
        if len(stafiles) == 0:
            continue

        staids = list(set([".".join(basename(x).split(".")[:2]) for x in stafiles]))
        part_func = partial(staevent, source=source)
        pool = mp.Pool(1)
        pool.starmap(part_func, zip(staids))
        pool.close()
        pool.join()
