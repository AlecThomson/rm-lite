#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     util_FITS.py                                                      #
#                                                                             #
# PURPOSE:  Utility functions to operate on FITS data.                        #
#                                                                             #
# MODIFIED: 19-November-2015 by C. Purcell                                    #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  mkWCSDict         ... parse fits header using wcslib                       #
#  strip_fits_dims   ... strip header and / or data dimensions                #
#  get_beam_from_header ... fetch the beam parameters from a FITS header      #
#  get_beam_area     ... calculate the effective beam area in px              #
#  get_subfits       ... cut a sub portion from a FITS cube                   #
#  create_simple_fits_hdu ... create a blank FITS HDU with full headers       #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 Cormac R. Purcell                                        #
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
import re
import sys

import astropy.io.fits as pf
import numpy as np


# -----------------------------------------------------------------------------#
def strip_fits_dims(data=None, header=None, minDim=2, forceCheckDims=0):
    """
    Strip array and / or header dimensions from a FITS data-array or header.
    """

    xydata = None

    # Strip unused dimensions from the header
    if data is not None:
        naxis = len(data.shape)
        extraDims = naxis - minDim
        if extraDims < 0:
            print("Too few dimensions in data. ")
            sys.exit(1)

        # Slice the data to strip the extra dims
        if extraDims == 0:
            xydata = data.copy()
        elif extraDims == 1:
            xydata = data[0].copy()
        elif extraDims == 2:
            xydata = data[0][0].copy()
        elif extraDims == 3:
            xydata = data[0][0][0].copy()
        else:
            print("Data array contains %s axes" % naxis)
            print("This script supports up to 5 axes only.")
            sys.exit(1)
        del data

    # Strip unused dimensions from the header
    if header is not None:
        header = header.copy()
        naxis = header["NAXIS"]

        stripHeadKeys = [
            "NAXIS",
            "CRVAL",
            "CRPIX",
            "CDELT",
            "CTYPE",
            "CROTA",
            "CD1_",
            "CD2_",
            "CUNIT",
        ]

        # Force a check on all relevant keywords
        if forceCheckDims > 0:
            for key in stripHeadKeys:
                for i in range(forceCheckDims + 1):
                    if key + str(i) in header:
                        if i > naxis:
                            naxis = i

        # Force a check on max dimensions of the PC keyword array
        if forceCheckDims > 0:
            for i in range(1, forceCheckDims + 1):
                for j in range(1, forceCheckDims + 1):
                    if naxis < max([i, j]):
                        naxis = max([i, j])

        extraDims = naxis - minDim
        if extraDims < 0:
            print("Too few dimensions in data. ")
            sys.exit(1)

        # Delete the entries
        for i in range(minDim + 1, naxis + 1):
            for key in stripHeadKeys:
                if key + str(i) in header:
                    try:
                        del header[key + str(i)]
                    except Exception:
                        pass

        # Delete the PC array keyword entries
        for i in range(1, naxis + 1):
            for j in range(1, naxis + 1):
                key = "PC" + "%03d" % i + "%03d" % j
                if i > minDim or j > minDim:
                    if key in header:
                        try:
                            del header[key]
                        except Exception:
                            pass

        header["NAXIS"] = minDim
        header["WCSAXES"] = minDim

    # Return the relevant object(s)
    if xydata is not None and header is not None:
        return [xydata, header]
    elif xydata is not None and header is None:
        return xydata
    elif xydata is None and header is not None:
        return header
    else:
        print("Both header and data are 'Nonetype'.")
        sys.exit(1)


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
