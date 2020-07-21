## This script contains example inputs for the atlas ETC.

########## IMPORTS #########

## Some general imports
import numpy as np
import sys

## ETC import
sys.path.append('../../atlas_etc')
from etc_code import atlas_etc, etc_plot



## INPUT 1: ATLAS Wide Survey ==========
# H-alpha emission at z=2.8 (observed 2.5 um) with flux of 5e-18 erg/s/cm2. Exposure time of 5000s, 5 exposures.
# Spatial extent of object is 0.7 arcsec, slit with (fixed) is 0.75 arcsec. 
input1 = {"band":"atlas_IR", # name of band
            "time":5000, # exposure time in seconds
            "slit_width":0.75,  # width of slit in arcseconds
            "Nreads":16, # number of reads in fowler
            "theta":0.7, # angular size of the object along the slit
            "nExp":5, # the number of exposures: if nEXP > 2, assumes a 2 point dither
            "lineF":5, # line flux in 10^-18 erg/s/cm^2
            "lineW":6563, # line wavelength of interest (rest or observed) in Ang/microns
            "FWHM":200, # rest-frame line FWHM in km/s
            "z":2.8, # redshift of line of interest
            "specFile":"none", # If not "none": filename of user spectrum (microns/Ang,F_nu). If "flat" assume flat f_nu spectrum. If "none" assume line
            "mag":-99, # magnitude in AB (-99 of not used)
            "NormalizeUserSpec":False, # if TRUE then user spectrum will be scaled to magnitude (mag)
            "InputInAngstroms":True, # True if user input wavelengths are in Angstroms
            "SN":-99 # desired signal to noise. If not -99, then this is used to calculate the exposure time
            }



## Run the ETC
#etc_output = atlas_etc(userinput=input1)

## Print some output
#print(etc_output["summary_struct"])

## Make some figures
#etc_plot(spec_struct=etc_output["spec_struct"], filename="input1.pdf")


## INPUT 2: ATLAS Wide Survey =============
# Constant (in f_nu) continuum normalized to 23AB (average over 1-4um). Exposure time of 5000s, 5 exposures.
# Spatial extent of object is 0.7 arcsec, slit with (fixed) is 0.75 arcsec. 
input2 = {"band":"atlas_IR", # name of band
            "time":5000, # exposure time in seconds
            "slit_width":0.75,  # width of slit in arcseconds
            "Nreads":16, # number of reads in fowler
            "theta":0.7, # angular size of the object along the slit
            "nExp":5, # the number of exposures: if nEXP > 2, assumes a 2 point dither
            "lineF":-99, # line flux in 10^-18 erg/s/cm^2
            "lineW":-99, # line wavelength of interest (rest or observed) in Ang/microns
            "FWHM":200, # rest-frame line FWHM in km/s
            "z":0, # redshift of line of interest
            "specFile":"flat", # If not "none": filename of user spectrum (microns/Ang,F_nu). If "flat" assume flat f_nu spectrum. If "none" assume line
            "mag":23, # magnitude in AB (-99 of not used)
            "NormalizeUserSpec":False, # if TRUE then user spectrum will be scaled to magnitude (mag)
            "InputInAngstroms":True, # True if user input wavelengths are in Angstroms
            "SN":-99 # desired signal to noise. If not -99, then this is used to calculate the exposure time
            }


## Run the ETC
etc_output = atlas_etc(userinput=input2)

## Print some output
print(etc_output["summary_struct"])

## Make some figures
etc_plot(spec_struct=etc_output["spec_struct"], filename="input2.pdf")

#####