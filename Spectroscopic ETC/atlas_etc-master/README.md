# ATLAS Exposure Time Calculator

Exposure Time Calculator (ETC) for the ATLAS project. The ETC is based on the MOSFIRE ETC and has been translated from IDL to Python.

## Installation

Simply clone the repository:
```
git clone https://github.com/afaisst/atlas_etc.git
```

The repository contains the following files and directories
- etc_code.py: This is the main program
- etc_data: Directory containing the telescope, background, and instrument data files
  - background: Directory containing the sky background files
  - filter: Directory containing the filter transmission files
  - spec_throughput: Directory containing the spectral throughput files
  - telescope: Directory containing the telescope stat files
- templates: Directory containing an example model template spectrum
- example: Directory containing an example script to run the ETC (including output in PDF format)

You also need to set an environment variable pointing to the ATLAS ETC installation path.
Add one of these lines to your shell initialization script (e.g., *~/.bashrc*, *~/.zshrc*, *~/.tshrc* or similar). The line you need to choose depends on your shell type.

```
export ATLAS_ETC_PATH=/Users/afaisst/Work/ATLAS/ETC/atlas_etc/
```

or

```
setenv ATLAS_ETC_PATH /Users/afaisst/Work/ATLAS/ETC/atlas_etc/
```

This installation path should include the directory "etc_data" with the sub-folder structure as downloaded from this repository.

To import the code to your Python script, use something like:

```
sys.path.append('my/path/atlas_etc')
from etc_code import *
```

You can use an absolute path (as in the above example) or a relative path (e.g., if in "example/" directory, you can use "../../atlas_etc").


## Telescope and Instrument Input Files

The ETC needs several input files that reflect the properties of the sky background, as well as telescope and instrument properties.

#### Telescope parameter file (main parameter file in etc_data/telescope/)
This is the main parameter file, containing the file names of the files listed below as well as other telescope and instrument related quantities.
The layout of this file is the following (from example input):

```
### Main Telescope parameter input file ###
## filename: telescope_data_atlas_IR.txt

## File Names (these files must be in the correct directories in the ATLAS ETC main directory)
bk_filename   backgrounds_mjy_per_str_ra_53.1_dec_-27.8.txt # File name of sky background file (in /etc_data/background/)
tp_filename   throughputSpec_atlas_IR.txt                   # File name of spectral throughput file (in /etc_data/spec_throughput/)
filt_filename filter_atlas_IR.txt                           # File name of filter transmission file (in /etc_data/filter/)

## linearity limits of detector
one_per_limit   55900 # one percent non-linearity limit [e-]
five_per_limit  79550 # five percent non-linearity limit [e-]
sat_limit       92450 # saturation limit [e-]

## Telescope constants
AT          17671 # Area of the telescope ((150/2.)**2 * np.pi) [cm2]
slit_W      0.75  # slit width to measure R_theta [arcsec]
pix_scale   0.39  # spatial pixel scale [arcsec/px]
tot_Npix    4096  # Number of pixels in dispersion direction on the detector
det_RN      5     # Detector readnoise (correlated double sampling) [e-/px]
Nmirror     2     # Number of mirrors

## Other stats
disp      10.0  # dispersion [angstrom/px]
lambda    2.5   # central wavelength [microns]
dark      0.005 # dark current [e-/s]
Mref      1.0   # Mirror reflectance (reflectance already included in spectroscopic throughput)
```

The keys can be in any order. Note that "band" (key in the user input, see below) currently links to the telescope file via "telescope_data_[band].txt"


#### Sky Background
The sky background is encoded in a text file ("etc_data/background/" relative to installation path). Currently, it has to have the wavelength (in microns) and the sky background (in mJy/sr).

#### Filter Transmission
The filter transmission curve is encoded in a text file ("etc_data/filter/" relative to installation path) and must contain wavelength (in microns) and transmission.

#### Spectral Throughput
The spectral throughput curve (i.e., including all mirror elements, and instrument losses, etc) is encoded in a text file ("etc_data/spec_throughput/" relative to installation path) and must contain the wavelength (in mircons) and the throughput.


## Usage

The ETC function can be called in Python simply by
```
from etc_code import *
output = atlas_etc(userinput)
```
where "userinput" is a Python dictionary (see below) and output is a dictionary with keys "summary_struct" and "spec_struct". The former contains important numbers (such as exposure time, average S/N). The later contains per-pixel quantities (such as the S/N spectrum) and can be fed to the *etc_plot()* function to visualize the results (see below).

The user input dictionary contains the following keys (with explanation):
- band: Name of band (currently linked to the telescope file via "telescope_data_[band].txt")
- time: Exposure time [s]. Note that if "SN" (signal-to-noise) is set, then the exposure time is used that corresponds to that S/N.
- slit_width: Slit width [arcsec]
- Nreads: Number of reads in fowler [int]
- theta: Angular size of the object along the slit [arcsec]
- nExp: Number of exposures [int]. Note that if "nExp" > 1, a 2-point dither is assumed
- lineF: Flux of single emission line [10^-18 erg/s/cm^2]
- lineW: Wavelength of single emission [mircons or Angstroms]
- FWHM: Rest-frame line FWHM [km/s]
- z: Redshift of line of interest or user spectrum [float]
- specFile: File name of user spectrum in (microns or Angstroms, F_nu) [string]. Set to "none" of not used and "flat" if flat spectrum (in f_nu).
- mag: Magnitude to scale user spectrum [AB]. Set to -99 if not used. 
- NormalizeUserSpec: Set to TRUE if user spectrum should be normalized to "mag" [True/False].
- InputInAngstroms: Set to TRUE of user spectrum wavelength or "lineW" is in Angstroms. If FALSE, microns are assumed [True/False]
- SN: Desired signal to noise to comput exposure time ("time" is not considered in that case) [float]. Set to -99 if not used.

There are three different kinds of inputs:
1. **A user spectrum.** Must have two column text file with columns (i) wavelength (microns or Angstroms) and (ii) Flux (in f_nu). This input is automatically assuemd if "specFile" is a path (meaning not "none" or "flat"). Use "NormalizeUserSpec" and "mag" to normalize the spectrum. A spectrum in observed frame is recommended, but you can use "z" to redshift the spectrum, meaning wavelength -> wavelength*(1+z) and flux -> flux/(1+z). The latter conserves the total flux.
2. **A specific line.** For this option, set "specFile" to "none" and specify the line flux ("lineF"), line central wavelength ("lineW"), and FWHM ("FWHM"). Use the redshift ("z") to shift the line (meaning lineW -> lineW*(1+z) ).
3. **A flat (in f_nu) spectrum.** This option is used when "specFile" is "flat". Use "mag" to normalized the flat spectrum (you don't have to set "NormalizeUserSpec" to TRUE).

## Visualization
The function
```
etc_plot(spec_struct,filename)
```
can be used to visualize the results. Specifically, to plot the transmissions (spectral and filter), sky background, noise level, input signal, and resulting S/N. The "spec_struct" is in the output dictionary of the *atlas_etc()* main function and can be directly used from there. The argument "filename" can be set up a path (e.g., "dir/dir/plot.pdf") to save the figure or set to "none" in which case the plot shows up but is not saved.


## Examples

An example code to run the ATLAS ETC is given in the "example/", called "etc_example.py". The script contains two example calculations for *(i)* a H-alpha line at z=2.8 with flux of 5e-18 erg/s/cm2 and *(ii)* a flat continuum scaled to a magnitude of 23AB (median 1-4 micro meters). The settings of the ATLAS Wide survey (5000s total exposure) is assuemd. The directory includes the output figures in PDF format ("input1.pdf" and "input2.pdf").

More examples (not included in example script) are listed below:

**Single H-alpha line at z=2.8 with flux equal to 5e-18 erg/s/cm2** 
```
userinput = {"band":"atlas_IR",
            "time":5000,
            "slit_width":0.75,
            "Nreads":16,
            "theta":0.7,
            "nExp":5,
            "lineF":5,
            "lineW":6563,
            "FWHM":200,
            "z":2.8,
            "specFile":"none",
            "mag":-99,
            "NormalizeUserSpec":False,
            "InputInAngstroms":True,
            "SN":-99,
            }
```

**Single H-alpha line at z=2.8, estimate exposure time for a given S/N=5 (integrated over the line)** 
```
userinput = {"band":"atlas_IR",
            "time":5000,
            "slit_width":0.75,
            "Nreads":16,
            "theta":0.7,
            "nExp":5,
            "lineF":5,
            "lineW":6563,
            "FWHM":200,
            "z":2.8,
            "specFile":"none",
            "mag":-99,
            "NormalizeUserSpec":False,
            "InputInAngstroms":True,
            "SN":5,
            }
```


**Example User Spectrum (included in repo) without normalization (K=19mag galaxy at z=2 with no dust attenuation and H-alpha flux of 5e-17 erg/s/cm2)**
```
userinput = {"band":"atlas_IR",
            "time":5000,
            "slit_width":0.75,
            "Nreads":16,
            "theta":0.7,
            "nExp":5,
            "lineF":-99,
            "lineW":-99,
            "FWHM":200,
            "z":0,
            "specFile":"./example_spec/Galaxy_Kmag_19_Haflux_5e-17_ebmv_0_z_2_highResSpec_obs_redshift.txt",
            "mag":-99,
            "NormalizeUserSpec":False,
            "InputInAngstroms":True,
            "SN":-99
            }
```

**Example User Spectrum (included in repo) normalized to 20AB magnitude over the filter**
```
userinput = {"band":"atlas_IR",
            "time":5000,
            "slit_width":0.75,
            "Nreads":16,
            "theta":0.7,
            "nExp":5,
            "lineF":-99,
            "lineW":-99,
            "FWHM":200,
            "z":0,
            "specFile":"./example_spec/Galaxy_Kmag_19_Haflux_5e-17_ebmv_0_z_2_highResSpec_obs_redshift.txt", 
            "mag":20,
            "NormalizeUserSpec":True,
            "InputInAngstroms":True,
            "SN":-99,
            }
```

**Flat spectrum (in f_nu) normalized to 20AB magnitude**
```
userinput = {"band":"atlas_IR",
            "time":5000,
            "slit_width":0.75,
            "Nreads":16,
            "theta":0.7,
            "nExp":5,
            "lineF":-99,
            "lineW":-99,
            "FWHM":200,
            "z":0,
            "specFile":"flat",
            "mag":20,
            "NormalizeUserSpec":False,
            "InputInAngstroms":True,
            "SN":-99,
            }
```

## To do list

The ETC works but it has to be modified to reflect the specifications of ATLAS
- Get a real transmission curve for ATLAS (tophat is currently assumed for filter and spectroscopic throughput).


