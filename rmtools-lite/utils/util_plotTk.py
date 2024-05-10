#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     util_plotTk.py                                                    #
#                                                                             #
# PURPOSE:  Plotting functions for the POSSUM pipeline                        #
#                                                                             #
# MODIFIED: 31-Jan-2018 by C. Purcell                                         #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
# xfloat                                                                      #
# xint                                                                        #
# filter_range_indx                                                           #
# tweakAxFormat                                                               #
# format_ticks                                                                #
# CustomNavbar                                                                #
# plot_I_vs_nu_ax                                                             #
# plot_PQU_vs_nu_ax                                                           #
# plot_rmsIQU_vs_nu_ax                                                        #
# plot_pqu_vs_lamsq_ax                                                        #
# plot_psi_vs_lamsq_ax                                                        #
# plot_q_vs_u_ax                                                              #
# plot_RMSF_ax                                                                #
# gauss                                                                       #
# plot_dirtyFDF_ax                                                            #
# plot_cleanFDF_ax                                                            #
# plot_hist4_ax                                                               #
#                                                                             #
# #-------------------------------------------------------------------------# #
#                                                                             #
# plotSpecIPQU                                                                #
# plotSpecRMS                                                                 #
# plotPolang                                                                  #
# plotFracPol                                                                 #
# plotFracQvsU                                                                #
# plot_Ipqu_spectra_fig                                                       #
# plotPolsummary                                                              #
# plotPolresidual                                                             #
# plot_rmsf_fdf_fig                                                           #
# plotRMSF                                                                    #
# plotDirtyFDF                                                                #
# plotCleanFDF                                                                #
# plotStampI                                                                  #
# plotStampP                                                                  #
# plotSctHstQuery                                                             #
# mk_hist_poly                                                                #
# label_format_exp                                                            #
# plot_complexity_fig                                                         #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 -2018 Cormac R. Purcell                                  #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
# =============================================================================#

import math as m
import os
import sys
import tkinter.ttk
import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

# Alter the default linewidths etc.
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["xtick.major.size"] = 8.0
mpl.rcParams["xtick.minor.size"] = 4.0
mpl.rcParams["ytick.major.size"] = 8.0
mpl.rcParams["ytick.minor.size"] = 4.0
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = 12.0

# Quick workaround to check if there is a chance for matplotlin to catch an
# X DISPLAY
try:
    if os.environ["rmsy_mpl_backend"] == "Agg":
        print('Environment variable rmsy_mpl_backend="Agg" detected.')
        print('Using matplotlib "Agg" backend in order to save the plots.')
        mpl.use("Agg")
except Exception:
    pass

# Constants
C = 2.99792458e8


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def tweakAxFormat(
    ax,
    pad=10,
    loc="upper right",
    linewidth=1,
    ncol=1,
    bbox_to_anchor=(1.00, 1.00),
    showLeg=True,
):
    # Axis/tic formatting
    ax.tick_params(pad=pad)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markeredgewidth(linewidth)

    # Legend formatting
    if showLeg:
        leg = ax.legend(
            numpoints=1,
            loc=loc,
            shadow=False,
            borderaxespad=0.3,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
        )
        for t in leg.get_texts():
            t.set_fontsize("small")
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_alpha(0.5)

    return ax


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
class CustomNavbar(NavigationToolbar2Tk):
    """Custom navigation toolbar subclass"""

    def __init__(self, canvas, window):
        NavigationToolbar2Tk.__init__(self, canvas, window)
        self.legStat = []
        for i in range(len(self.canvas.figure.axes)):
            ax = self.canvas.figure.axes[i]
            if ax.get_legend() is None:
                self.legStat.append(None)
            else:
                self.legStat.append(True)

    def _init_toolbar(self):
        NavigationToolbar2Tk._init_toolbar(self)

        # Add the legend toggle button
        self.legBtn = tkinter.ttk.Button(
            self, text="Hide Legend", command=self.toggle_legend
        )
        self.legBtn.pack(side="left")

        # Remove the back and forward buttons
        # List of buttons is in self.toolitems
        buttonLst = self.pack_slaves()
        buttonLst[1].pack_forget()
        buttonLst[2].pack_forget()

    def toggle_legend(self):
        for i in range(len(self.canvas.figure.axes)):
            ax = self.canvas.figure.axes[i]
            if self.legStat[i] is not None:
                ax.get_legend().set_visible(not self.legStat[i])
                self.legStat[i] = not self.legStat[i]
        if self.legBtn["text"] == "Hide Legend":
            self.legBtn["text"] = "Show Legend"
        else:
            self.legBtn["text"] = "Hide Legend"
        self.canvas.draw()


# -----------------------------------------------------------------------------#
def plot_I_vs_nu_ax(
    ax,
    freqArr_Hz,
    IArr,
    dIArr=None,
    freqHirArr_Hz=None,
    IModArr=None,
    axisYright=False,
    axisXtop=False,
    units="",
):
    """Plot the I spectrum and an optional model."""

    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if freqHirArr_Hz is None:
        freqHirArr_Hz = freqArr_Hz

    # Plot I versus frequency
    ax.errorbar(
        x=freqArr_Hz / 1e9,
        y=IArr,
        yerr=dIArr,
        mfc="none",
        ms=2,
        fmt="D",
        mec="k",
        ecolor="k",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes I",
    )

    # Plot the model
    if IModArr is not None:
        ax.plot(freqHirArr_Hz / 1e9, IModArr, color="tab:red", lw=2.5, label="I Model")

    # Scaling & formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = (np.nanmax(freqArr_Hz) - np.nanmin(freqArr_Hz)) / 1e9
    ax.set_xlim(
        np.nanmin(freqArr_Hz) / 1e9 - xRange * 0.05,
        np.nanmax(freqArr_Hz) / 1e9 + xRange * 0.05,
    )
    ax.set_xlabel(r"$\nu$ (GHz)")
    ax.set_ylabel(rf"Flux Density ({units})")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def plot_pqu_vs_lamsq_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    pArr=None,
    dqArr=None,
    duArr=None,
    dpArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if lamSqHirArr_m2 is None:
        lamSqHirArr_m2 = lamSqArr_m2

    # Calculate p and errors
    if pArr is None:
        pArr = np.sqrt(qArr**2.0 + uArr**2.0)
        pArr = np.where(np.isfinite(pArr), pArr, np.nan)
    if dpArr is None:
        if dqArr is None or duArr is None:
            dpArr = None
        else:
            dpArr = np.sqrt(dqArr**2.0 + duArr**2.0)
            dpArr = np.where(np.isfinite(dpArr), dpArr, np.nan)

    # Plot p, q, u versus lambda^2
    #    """
    ax.errorbar(
        x=lamSqArr_m2,
        y=qArr,
        yerr=dqArr,
        mec="tab:blue",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="tab:blue",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes q",
    )
    ax.errorbar(
        x=lamSqArr_m2,
        y=uArr,
        yerr=duArr,
        mec="tab:red",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="tab:red",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Stokes u",
    )
    ax.errorbar(
        x=lamSqArr_m2,
        y=pArr,
        yerr=dpArr,
        mec="k",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="k",
        alpha=0.5,
        elinewidth=1.0,
        capsize=2,
        label="Intensity p",
    )
    """
    ax.errorbar(x=lamSqArr_m2, y=pArr, yerr=dpArr, mec='k', mfc='tab:red', ms=2,
                fmt='D', ecolor='k', elinewidth=1.0, capsize=2,
                label='Intensity p')

    ax.errorbar(x=lamSqArr_m2, y=qArr, yerr=dqArr, mec='tab:blue', mfc='tab:red', ms=2,
                fmt='D', ecolor='tab:blue', elinewidth=1.0, capsize=2,
                label='Stokes q')

    ax.errorbar(x=lamSqArr_m2, y=uArr, yerr=duArr, mec='tab:red', mfc='tab:blue', ms=2,
                fmt='D', ecolor='tab:red', elinewidth=1.0, capsize=2,
                label='Stokes u')
    """

    # Plot the models
    if qModArr is not None:
        ax.plot(
            lamSqHirArr_m2, qModArr, color="tab:blue", alpha=1, lw=0.1, label="Model q"
        )
    if uModArr is not None:
        ax.plot(
            lamSqHirArr_m2, uModArr, color="tab:red", alpha=1, lw=0.1, label="Model u"
        )
    if qModArr is not None and uModArr is not None:
        pModArr = np.sqrt(qModArr**2.0 + uModArr**2.0)
        ax.plot(lamSqHirArr_m2, pModArr, color="k", alpha=1, lw=0.1, label="Model p")
    if model_dict is not None:
        errDict = {}
        QUerrmodel = []
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel.append(model_dict["model"](errDict, lamSqHirArr_m2))
        QUerrmodel = np.array(QUerrmodel)
        low_re, med_re, high_re = np.percentile(
            np.real(QUerrmodel), [16, 50, 84], axis=0
        )
        low_im, med_im, high_im = np.percentile(
            np.imag(QUerrmodel), [16, 50, 84], axis=0
        )
        low_abs, med_abs, high_abs = np.percentile(
            np.abs(QUerrmodel), [16, 50, 84], axis=0
        )

        ax.plot(lamSqHirArr_m2, med_re, "-", color="tab:blue", linewidth=0.1, alpha=1)
        ax.fill_between(lamSqHirArr_m2, low_re, high_re, color="tab:blue", alpha=0.5)
        ax.plot(lamSqHirArr_m2, med_im, "-", color="tab:red", linewidth=0.1, alpha=1)
        ax.fill_between(lamSqHirArr_m2, low_im, high_im, color="tab:red", alpha=0.5)
        if qModArr is not None and uModArr is not None:
            ax.plot(lamSqHirArr_m2, med_abs, "-", color="k", linewidth=0.1, alpha=1)
            ax.fill_between(lamSqHirArr_m2, low_abs, high_abs, color="k", alpha=0.5)

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(lamSqArr_m2) - np.nanmin(lamSqArr_m2)
    ax.set_xlim(
        np.nanmin(lamSqArr_m2) - xRange * 0.05, np.nanmax(lamSqArr_m2) + xRange * 0.05
    )
    yDataMax = max(np.nanmax(pArr), np.nanmax(qArr), np.nanmax(uArr))
    yDataMin = min(np.nanmin(pArr), np.nanmin(qArr), np.nanmin(uArr))
    yRange = yDataMax - yDataMin
    medErrBar = np.max(
        [float(nanmedian(dpArr)), float(nanmedian(dqArr)), float(nanmedian(duArr))]
    )
    ax.set_ylim(
        yDataMin - 2 * medErrBar - yRange * 0.05,
        yDataMax + 2 * medErrBar + yRange * 0.1,
    )
    ax.set_xlabel(r"$\lambda^2$ (m$^2$)")
    ax.set_ylabel("Fractional Polarisation")
    ax.axhline(0, linestyle="--", color="grey")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax)


# -----------------------------------------------------------------------------#
def plot_psi_vs_lamsq_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    dqArr=None,
    duArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Default to non-high-resolution inputs
    if lamSqHirArr_m2 is None:
        lamSqHirArr_m2 = lamSqArr_m2

    # Calculate the angle and errors
    pArr = np.sqrt(qArr**2.0 + uArr**2.0)
    psiArr_deg = np.degrees(np.arctan2(uArr, qArr) / 2.0)
    if dqArr is None or duArr is None:
        dQUArr = None
        dPsiArr_deg = None
    else:
        dQUArr = np.sqrt(dqArr**2.0 + duArr**2.0)
        dPsiArr_deg = np.degrees(
            np.sqrt((qArr * duArr) ** 2.0 + (uArr * dqArr) ** 2.0) / (2.0 * pArr**2.0)
        )

    # Plot psi versus lambda^2
    ax.errorbar(
        x=lamSqArr_m2,
        y=psiArr_deg,
        yerr=dPsiArr_deg,
        mec="k",
        mfc="none",
        ms=2,
        fmt="D",
        ecolor="k",
        alpha=0.3,
        elinewidth=1.0,
        capsize=2,
    )

    # Plot the model
    if qModArr is not None and uModArr is not None:
        psiHirArr_deg = np.degrees(np.arctan2(uModArr, qModArr) / 2.0)
        ax.plot(
            lamSqHirArr_m2,
            psiHirArr_deg,
            color="tab:red",
            lw=0.1,
            label=r"Model $\psi$",
        )
    if model_dict is not None:
        errDict = {}
        psi_errmodel = []
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel = model_dict["model"](errDict, lamSqHirArr_m2)
            Qerrmodel = np.real(QUerrmodel)
            Uerrmodel = np.imag(QUerrmodel)
            psi_errmodel.append(np.degrees(np.arctan2(Uerrmodel, Qerrmodel) / 2.0))

        psi_errmodel = np.array(psi_errmodel)
        low, med, high = np.percentile(psi_errmodel, [16, 50, 84], axis=0)

        ax.plot(lamSqHirArr_m2, med, "-", color="tab:red", linewidth=0.1, alpha=1)
        ax.fill_between(
            lamSqHirArr_m2, low, high, color="tab:red", linewidth=0.1, alpha=0.5
        )

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(lamSqArr_m2) - np.nanmin(lamSqArr_m2)
    ax.set_xlim(
        np.nanmin(lamSqArr_m2) - xRange * 0.05, np.nanmax(lamSqArr_m2) + xRange * 0.05
    )
    ax.set_ylim(-99.9, 99.9)
    ax.set_xlabel(r"$\lambda^2$ (m$^2$)")
    ax.set_ylabel(r"$\psi$ (degrees)")
    ax.axhline(0, linestyle="--", color="grey")
    ax.minorticks_on()

    # Format tweaks
    ax = tweakAxFormat(ax, showLeg=False)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
def plot_q_vs_u_ax(
    ax,
    lamSqArr_m2,
    qArr,
    uArr,
    dqArr=None,
    duArr=None,
    lamSqHirArr_m2=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    axisYright=False,
    axisXtop=False,
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot u versus q
    ax.errorbar(
        x=qArr,
        y=uArr,
        xerr=dqArr,
        yerr=duArr,
        mec="grey",
        mfc="none",
        ms=1,
        fmt=".",
        ecolor="grey",
        elinewidth=1.0,
        capsize=2,
        zorder=1,
    )
    freqArr_Hz = C / np.sqrt(lamSqArr_m2)
    ax.scatter(
        x=qArr,
        y=uArr,
        c=freqArr_Hz,
        cmap="rainbow_r",
        s=30,
        marker="D",
        edgecolor="none",
        linewidth=0.1,
        zorder=2,
    )

    # Plot the model
    if qModArr is not None and uModArr is not None:
        ax.plot(qModArr, uModArr, color="k", lw=0.1, label="Model q & u", zorder=2)
    if model_dict is not None:
        errDict = {}
        # Sample the posterior randomly 100 times
        for i in range(1000):
            idx = np.random.choice(np.arange(model_dict["posterior"].shape[0]))
            for j, name in enumerate(model_dict["parNames"]):
                errDict[name] = model_dict["posterior"][name][idx]
            QUerrmodel = model_dict["model"](errDict, lamSqHirArr_m2)
            Qerrmodel = np.real(QUerrmodel)
            Uerrmodel = np.imag(QUerrmodel)
            ax.plot(Qerrmodel, Uerrmodel, color="k", lw=0.1, alpha=0.5, zorder=2)

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(qArr) - np.nanmin(qArr)
    ax.set_xlim(np.nanmin(qArr) - xRange * 0.05, np.nanmax(qArr) + xRange * 0.05)
    yRange = np.nanmax(uArr) - np.nanmin(uArr)
    ax.set_ylim(np.nanmin(uArr) - yRange * 0.05, np.nanmax(uArr) + yRange * 0.05)
    ax.set_xlabel("Stokes q")
    ax.set_ylabel("Stokes u")
    ax.axhline(0, linestyle="--", color="grey")
    ax.axvline(0, linestyle="--", color="grey")
    ax.axis("equal")
    ax.minorticks_on()
    # Format tweaks
    ax = tweakAxFormat(ax, showLeg=False)


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def plot_RMSF_ax(
    ax, phiArr, RMSFArr, fwhmRMSF=None, axisYright=False, axisXtop=False, doTitle=False
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the RMSF
    ax.step(phiArr, RMSFArr.real, where="mid", color="tab:blue", lw=0.5, label="Real")
    ax.step(
        phiArr, RMSFArr.imag, where="mid", color="tab:red", lw=0.5, label="Imaginary"
    )
    ax.step(phiArr, np.abs(RMSFArr), where="mid", color="k", lw=1.0, label="PI")
    ax.axhline(0, color="grey")
    if doTitle:
        ax.text(0.05, 0.84, "RMSF", transform=ax.transAxes)

    # Plot the Gaussian fit
    if fwhmRMSF is not None:
        yGauss = gauss([1.0, 0.0, fwhmRMSF])(phiArr)
        ax.plot(
            phiArr,
            yGauss,
            color="g",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Gaussian",
            lw=2.0,
            ls="--",
        )

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Normalised Units")
    ax.set_xlabel(r"$\phi$ rad m$^{-2}$")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------
def gauss(p):
    """Return a fucntion to evaluate a Gaussian with parameters
    p = [amp, mean, FWHM]"""

    a, b, w = p
    gfactor = 2.0 * m.sqrt(2.0 * m.log(2.0))
    s = w / gfactor

    def rfunc(x):
        y = a * np.exp(-((x - b) ** 2.0) / (2.0 * s**2.0))
        return y

    return rfunc


# -----------------------------------------------------------------------------#
def plot_dirtyFDF_ax(
    ax,
    phiArr,
    FDFArr,
    gaussParm=[],
    vLine=None,
    title="Dirty FDF",
    axisYright=False,
    axisXtop=False,
    doTitle=False,
    units="",
):
    # Set the axis positions
    if axisYright:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axisXtop:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    # Plot the FDF
    FDFpiArr = np.sqrt(np.power(FDFArr.real, 2.0) + np.power(FDFArr.imag, 2.0))
    ax.step(phiArr, FDFArr.real, where="mid", color="tab:blue", lw=0.5, label="Real")
    ax.step(
        phiArr, FDFArr.imag, where="mid", color="tab:red", lw=0.5, label="Imaginary"
    )
    ax.step(phiArr, FDFpiArr, where="mid", color="k", lw=1.0, label="PI")
    if doTitle == True:
        ax.text(0.05, 0.84, "Dirty FDF", transform=ax.transAxes)

    # Plot the Gaussian peak
    if len(gaussParm) == 3:
        # [amp, mean, FWHM]
        phiTrunkArr = np.where(
            phiArr >= gaussParm[1] - gaussParm[2] / 3.0, phiArr, np.nan
        )
        phiTrunkArr = np.where(
            phiArr <= gaussParm[1] + gaussParm[2] / 3.0, phiTrunkArr, np.nan
        )
        yGauss = gauss(gaussParm)(phiTrunkArr)
        ax.plot(
            phiArr,
            yGauss,
            color="magenta",
            marker="None",
            mfc="w",
            mec="g",
            ms=10,
            label="Peak Fit",
            lw=2.5,
            ls="-",
        )

    # Plot a vertical line
    if vLine:
        ax.axvline(vLine, color="magenta", ls="--", linewidth=1.5)

    # Scaling
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(8))
    xRange = np.nanmax(phiArr) - np.nanmin(phiArr)
    ax.set_xlim(np.nanmin(phiArr) - xRange * 0.01, np.nanmax(phiArr) + xRange * 0.01)
    ax.set_ylabel("Flux Density (" + units + ")")
    ax.set_xlabel(r"$\phi$ (rad m$^{-2}$)")
    ax.axhline(0, color="grey")

    # Format tweaks
    ax = tweakAxFormat(ax)
    ax.autoscale_view(True, True, True)


# -----------------------------------------------------------------------------#
# ax.autoscale_view(True,True,True)


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# Axis code above
# =============================================================================#
# Figure code below


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def plot_Ipqu_spectra_fig(
    freqArr_Hz,
    IArr,
    qArr,
    uArr,
    dIArr=None,
    dqArr=None,
    duArr=None,
    freqHirArr_Hz=None,
    IModArr=None,
    qModArr=None,
    uModArr=None,
    model_dict=None,
    fig=None,
    units="",
):
    """Plot the Stokes I, Q/I & U/I spectral summary plots."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(12.0, 8))

    # Default to non-high-resolution inputs
    if freqHirArr_Hz is None:
        freqHirArr_Hz = freqArr_Hz
    lamSqArr_m2 = np.power(C / freqArr_Hz, 2.0)
    lamSqHirArr_m2 = np.power(C / freqHirArr_Hz, 2.0)

    # Plot I versus nu axis
    ax1 = fig.add_subplot(221)
    plot_I_vs_nu_ax(
        ax=ax1,
        freqArr_Hz=freqArr_Hz,
        IArr=IArr,
        dIArr=dIArr,
        freqHirArr_Hz=freqHirArr_Hz,
        IModArr=IModArr,
        axisXtop=True,
        units=units,
    )

    # Plot p, q, u versus lambda^2 axis
    ax2 = fig.add_subplot(223)
    plot_pqu_vs_lamsq_ax(
        ax=ax2,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
    )

    # Plot psi versus lambda^2 axis
    ax3 = fig.add_subplot(222, sharex=ax2)
    plot_psi_vs_lamsq_ax(
        ax=ax3,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
        axisYright=True,
        axisXtop=True,
    )

    # Plot q versus u axis
    ax4 = fig.add_subplot(224)
    plot_q_vs_u_ax(
        ax=ax4,
        lamSqArr_m2=lamSqArr_m2,
        qArr=qArr,
        uArr=uArr,
        dqArr=dqArr,
        duArr=duArr,
        lamSqHirArr_m2=lamSqHirArr_m2,
        qModArr=qModArr,
        uModArr=uModArr,
        model_dict=model_dict,
        axisYright=True,
    )

    # Adjust subplot spacing
    fig.subplots_adjust(
        left=0.1, bottom=0.08, right=0.90, top=0.92, wspace=0.08, hspace=0.08
    )

    fig.tight_layout()

    return fig


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def plot_rmsf_fdf_fig(
    phiArr,
    FDF,
    phi2Arr,
    RMSFArr,
    fwhmRMSF=None,
    gaussParm=[],
    vLine=None,
    fig=None,
    units="flux units",
):
    """Plot the RMSF and FDF on a single figure."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(12.0, 8))
    # Plot the RMSF
    ax1 = fig.add_subplot(211)
    plot_RMSF_ax(
        ax=ax1, phiArr=phi2Arr, RMSFArr=RMSFArr, fwhmRMSF=fwhmRMSF, doTitle=True
    )
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.set_xlabel("")

    # Plot the FDF
    # Why are these next two lines here? Removing as part of units fix.
    #    if len(gaussParm)==3:
    #        gaussParm[0] *= 1e3
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_dirtyFDF_ax(
        ax=ax2,
        phiArr=phiArr,
        FDFArr=FDF,
        gaussParm=gaussParm,
        vLine=vLine,
        doTitle=True,
        units=units,
    )

    return fig


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def plot_complexity_fig(
    xArr,
    qArr,
    dqArr,
    sigmaAddqArr,
    chiSqRedqArr,
    probqArr,
    uArr,
    duArr,
    sigmaAdduArr,
    chiSqReduArr,
    probuArr,
    mDict,
    med=0.0,
    noise=1.0,
    fig=None,
):
    """Create the residual Stokes q and u complexity plots."""

    # Default to a pyplot figure
    if fig == None:
        fig = plt.figure(facecolor="w", figsize=(16.0, 8.0))

    # Plot the data and the +/- 1-sigma levels
    ax1 = fig.add_subplot(231)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.set_major_locator(MaxNLocator(7))
    ax1.errorbar(
        x=xArr,
        y=qArr,
        yerr=dqArr,
        ms=3,
        color="tab:blue",
        fmt="o",
        alpha=0.5,
        capsize=0,
    )
    ax1.errorbar(
        x=xArr, y=uArr, yerr=duArr, ms=3, color="tab:red", fmt="o", alpha=0.5, capsize=0
    )
    ax1.axhline(med, color="grey", zorder=10)
    ax1.axhline(1.0, color="k", linestyle="--", zorder=10)
    ax1.axhline(-1.0, color="k", linestyle="--", zorder=10)
    ax1.set_xlabel(r"$\lambda^2$")
    ax1.set_ylabel("Normalised Residual")

    # Plot the histogram of the data overlaid by the normal distribution
    H = 1.0 / np.sqrt(2.0 * np.pi * noise**2.0)
    xNorm = np.linspace(med - 3 * noise, med + 3 * noise, 1000)
    yNorm = H * np.exp(-0.5 * ((xNorm - med) / noise) ** 2.0)
    fwhm = noise * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ax2 = fig.add_subplot(232)
    ax2.tick_params(labelbottom="off")
    nBins = 15
    yMin = np.nanmin([np.nanmin(qArr), np.nanmin(uArr)])
    yMax = np.nanmax([np.nanmax(qArr), np.nanmax(uArr)])
    n, b, p = ax2.hist(
        qArr,
        nBins,
        range=(yMin, yMax),
        density=1,
        histtype="step",
        color="tab:blue",
        linewidth=1.0,
    )
    ax2.plot(xNorm, yNorm, color="k", linestyle="--", linewidth=1.5)
    n, b, p = ax2.hist(
        uArr,
        nBins,
        range=(yMin, yMax),
        density=1,
        histtype="step",
        color="tab:red",
        linewidth=1.0,
    )
    ax2.axvline(med, color="grey", zorder=11)
    ax2.set_title(r"Distribution of Data vs Normal")
    ax2.set_ylabel(r"Normalised Counts")

    # Plot the ECDF versus a normal CDF
    nData = len(xArr)
    ecdfArr = np.array(list(range(nData))) / float(nData)
    qSrtArr = np.sort(qArr)
    uSrtArr = np.sort(uArr)
    ax3 = fig.add_subplot(235, sharex=ax2)
    ax3.step(qSrtArr, ecdfArr, where="mid", color="tab:blue")
    ax3.step(uSrtArr, ecdfArr, where="mid", color="tab:red")
    x, y = norm_cdf(mean=med, std=noise, N=1000)
    ax3.plot(x, y, color="k", linewidth=1.5, linestyle="--", zorder=1)
    ax3.set_ylim(0, 1.05)
    ax3.axvline(med, color="grey", zorder=11)
    ax3.set_xlabel(r"Normalised Residual")
    ax3.set_ylabel(r"Normalised Counts")

    # Plot reduced chi-squared
    ax4 = fig.add_subplot(234)
    ax4.step(
        x=sigmaAddqArr, y=chiSqRedqArr, color="tab:blue", linewidth=1.0, where="mid"
    )
    ax4.step(
        x=sigmaAddqArr, y=chiSqReduArr, color="tab:red", linewidth=1.0, where="mid"
    )
    ax4.axhline(1.0, color="k", linestyle="--")
    ax4.set_xlabel(r"$\sigma_{\rm add}$")
    ax4.set_ylabel(r"$\chi^2_{\rm reduced}$")

    # Plot the probability distribution function
    ax5 = fig.add_subplot(233)
    ax5.tick_params(labelbottom="off")
    ax5.step(
        x=sigmaAddqArr,
        y=probqArr,
        linewidth=1.0,
        where="mid",
        color="tab:blue",
        alpha=0.5,
    )
    ax5.step(
        x=sigmaAdduArr,
        y=probuArr,
        linewidth=1.0,
        where="mid",
        color="tab:red",
        alpha=0.5,
    )
    ax5.axvline(mDict["sigmaAddQ"], color="tab:blue", linestyle="-", linewidth=1.5)
    ax5.axvline(
        mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(
        mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(mDict["sigmaAddU"], color="tab:red", linestyle="-", linewidth=1.5)
    ax5.axvline(
        mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.axvline(
        mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax5.set_title("Likelihood Distribution")
    ax5.set_ylabel(r"P($\sigma_{\rm add}$|data)")

    # Plot the CPDF
    CPDFq = np.cumsum(probqArr) / np.sum(probqArr)
    CPDFu = np.cumsum(probuArr) / np.sum(probuArr)
    ax6 = fig.add_subplot(236, sharex=ax5)
    ax6.step(x=sigmaAddqArr, y=CPDFq, linewidth=1.0, where="mid", color="tab:blue")
    ax6.step(x=sigmaAdduArr, y=CPDFu, linewidth=1.0, where="mid", color="tab:red")
    ax6.set_ylim(0, 1.05)
    ax6.axhline(0.5, color="grey", linestyle="-", linewidth=1.5)
    ax6.axvline(mDict["sigmaAddQ"], color="tab:blue", linestyle="-", linewidth=1.5)
    ax6.axvline(
        mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(
        mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(mDict["sigmaAddU"], color="tab:red", linestyle="-", linewidth=1.5)
    ax6.axvline(
        mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.axvline(
        mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    ax6.set_xlabel(r"$\sigma_{\rm add}$")
    ax6.set_ylabel(r"Cumulative Likelihood")

    # Zoom in
    xLim1 = np.nanmin(
        [
            mDict["sigmaAddQ"] - mDict["dSigmaAddMinusQ"] * 4.0,
            mDict["sigmaAddU"] - mDict["dSigmaAddMinusU"] * 4.0,
        ]
    )
    xLim1 = max(0.0, xLim1)
    xLim2 = np.nanmax(
        [
            mDict["sigmaAddQ"] + mDict["dSigmaAddPlusQ"] * 4.0,
            mDict["sigmaAddU"] + mDict["dSigmaAddPlusU"] * 4.0,
        ]
    )
    ax6.set_xlim(xLim1, xLim2)

    # Show the figure
    fig.subplots_adjust(
        left=0.07, bottom=0.09, right=0.97, top=0.92, wspace=0.25, hspace=0.05
    )

    return fig
