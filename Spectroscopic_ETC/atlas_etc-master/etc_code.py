## This is the main code for the ETC calculator

########## IMPORTS #########

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn

# ----- Set up plotting parameters -----
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

###### HELPER FUNCTIONS #######

## Read parameter file
# These files have the "SExtractor-type" format (key val comment)
# Returns a dictionary with key and value
def read_par_file(filename):
    '''
    This function reads the telescope parameter file and outputs a dictionary with (key,value). \\
    USAGE: read_par_file(filename) where filename is the name of the parameter file to read
    '''
    with open(filename , "r") as f:
        lines = f.readlines()
    

    # replace tab and line break
    lines = [line.replace("\t","  ").replace("\n", "") for line in lines ] 

    # get lines that are not empty or commented out
    lines = [ line for line in lines if line != "" if line[0] != "#" ] 


    # extract key, val, comment (if exist)
    extract = dict()
    for line in lines:
        try:
            key = line.split()[0]
            val = line.split()[1]
            try: # check if the value can be interpreted as float
                val = float(line.split()[1])
            except: # if not, make it a string
                val = str(line.split()[1])
            extract[key] = val
        except:
            print("Cannot interpret/read the line %s" & line)
            quit()
    
    return(extract)


## Degrade any spectrum to the spectral resolution of the instrument
## Input:
# in_wave: input wavelength (in microns)
# in_flux: input flux (f_nu)
# center_wave: center wavelength of filter (in microns)
# R: real resolution of spectrograph
# disp: dispersion (A/px)
#
## Output
# dictionary with (lam and flux)
#
def degrade_resolution(in_wave, in_flux , center_wave , R, disp, tot_Npix):
    '''
    Degrades an input array to the telescope resolution. \\ 
    USAGE: degrade_resolution(in_wave , in_flux , center_wave , R , disp) where \\
        in_wave: wavelength \\
        in_flux: flux or transmission \\
        center_wave: center wavelength \\
        R: Spectral resolution \\
        disp: dispersion [angstrom/px] \\
    OUTPUT: Dictionary including the degraded wavelength vs. flux: (lam,flux)
    '''

    # Number of pixels to be output - 50%
    # more than are on the detector to
    # cover the K band
    Npix_spec=tot_Npix * 3./2.

    #the speed of light in cm/s
    c=np.log10(29979245800.)

    # make a "velocity" grid centered at
    # the central wavelength of the band
    # sampled at 1 km/s
    vel=(np.arange(600001)-300000)
    in_vel=(in_wave/center_wave-1)*10.**(1*c-5)

    # create vectors in velocity space
    in_vel_short = in_vel[ np.where( (in_vel > vel[0]) & (in_vel < vel[600000]) )[0] ]
    in_flux_short = in_flux[ np.where( (in_vel > vel[0]) & (in_vel < vel[600000]) )[0] ]

    #interp_flux = np.interp(vel, in_flux_short, in_vel_short)
    interp_flux = np.interp(vel, in_vel_short, in_flux_short)

    #sigma  = the resolution of the spectrograph
    sigma = (10.**(c-5)/R)/(2*np.sqrt(2*np.log(2)))


    # make a smaller velocity array with
    # the same "resolution" as the steps in
    # vel, above
    n = round(8.*sigma)
    if (n % 2 == 0):
        n = n + 1
    vel_kernel = np.arange(n) - np.floor(n/2.0)

    # a gaussian of unit area and width sigma
    gauss_kernel = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*vel_kernel**2.0/sigma**2.0)

    # convolve flux with gaussian kernel
    convol_flux = np.convolve(interp_flux, gauss_kernel , mode="same") 
    convol_wave = center_wave * (vel*10.**(-1*c+5.0) + 1.0 )

    # and the real pixel scale of mosfire
    real_wave = np.arange(Npix_spec) * disp * 10.**(-4.)
    real_wave = real_wave - real_wave[int(np.round(Npix_spec/2.))]   
    real_wave = real_wave + center_wave 

    # interpolate onto the pixel scale of the detector
    out_wave = real_wave
    out_flux = np.interp(real_wave , convol_wave, convol_flux)
    
    out = {"lam": out_wave,
          "flux": out_flux}
    
    return(out)


## Simple tophat convolution
# res and res_new has to be in the same units (e.g., angstroms)
def tophat_convolve(y,res,res_new):
    n = np.round(res_new/res)
    if (n % 2 == 0):
        n = n + 1
    kk = np.arange(n)-np.floor(n/2.)
    kk = kk * 0.0 + 1
    kk[0] = 0
    kk[int(n-1)] = 0
    y = np.convolve(y, kk , mode="same")
    return(y)

###############################

#### MAIN PROGRAM ########

def atlas_etc(userinput):
    '''
    Main ATLAS ETC function. \\
    USAGE: atlas_etc(userinput) \\
    OUTPUT: This function returns a dictionary with keys "summary_struct" (containing some numbers such as average S/N, etc) and "spec_struct" (containing S/N, spectrum, etc) \\ \\
    Example User Input: 
        userinput = {"band":"NIR", # name of band
            "time":2*3600, # exposure time in seconds 
            "slit_width":2.8,  # width of slit in arcseconds 
            "Nreads":16, # number of reads in fowler 
            "theta":1.4, # angular size of the object along the slit 
            "nExp":2, # the number of exposures: if nEXP > 2, assumes a 2 point dither 
            "lineF":300, # line flux in 10^-18 erg/s/cm^2; set -99 if not used 
            "lineW":6563, # line wavelength of interest (rest or observed); set -99 if not used 
            "FWHM":30, # rest-frame line FWHM in km/s 
            "z":1.5, # redshift of line of interest 
            "specFile":"none", # If not "none": filename of user spectrum (microns,F_nu). If "flat" flat spectrum in f_nu. If "none", a line (lineF, lineW, FWHM) is used. 
            "mag":-99, # magnitude in AB (-99 of not used) 
            "NormalizeUserSpec":False, # if TRUE then user spectrum will be scaled to magnitude (mag) 
            "InputInAngstroms":True, # True if user input wavelengths are in Angstroms 
            "SN":-99, # desired signal to noise. If not -99, then this is used to calculate the exposure time 
        }
    '''


    ### ============= DITHER ======================

    ## if the number of exposures is greater than 1, assume two dither positions
    if userinput["nExp"] > 1:
        dither = 2.0
    else:
        dither = 1.0
    
    ### ============= DIRECTORY STRUCTURE AND NAMES ======================

    ## Directory structure
    # Make sure to have the environment variable "ATLAS_ETC_PATH" set.
    try:
        if os.environ["ATLAS_ETC_PATH"] == "none":
            pass
    except:
        print("Environment variable 'ATLAS_ETC_PATH' does not seem to be set. Add it to your shell initialization script and try again. See manual for more details.")
        quit()
    
    DIRS = dict()
    DIRS["bk_path"] = "%s/etc_data/background/" % os.environ["ATLAS_ETC_PATH"] # path to background data files
    DIRS["tp_path"] = "%s/etc_data/spec_throughput/" % os.environ["ATLAS_ETC_PATH"] # path to spectral throughput data files
    DIRS["filt_path"] = "%s/etc_data/filter/" % os.environ["ATLAS_ETC_PATH"] # path to filter transmission data files
    DIRS["tel_path"] = "%s/etc_data/telescope/" % os.environ["ATLAS_ETC_PATH"] # path to telescope data files

    ### ============= TELESCOPE DEFINITIONS AND OTHER CONSTANTS ======================
    ## The file should contain
    # - linearity limits
    # - telescope stats
    # - other stats (mirrors reflectivity, dark current, etc)
    try:
        telstats = read_par_file(os.path.join(DIRS["tel_path"] , "telescope_data_%s.txt" % userinput["band"]) )
    except:
        print("Something went wrong when reading the telescope parameter file. Check file name!")
        quit()

    # Add R-theta product: multiply by slit width to get resolution
    telstats["rt"] = 1000 * telstats["slit_W"]

    ## Load Filters Transmission Curve ---------------
    try:
        filter_trans = ascii.read(os.path.join(DIRS["filt_path"],telstats["filt_filename"]),names=["lam","trans"])
    except:
        print("Something went wrong when reading the filter transmission file. Check file name!")
        quit()


    ## Load Spectral Transmission Curve --------------
    try:
        throughput = ascii.read(os.path.join(DIRS["tp_path"] , telstats["tp_filename"]) , names=["tp_wave","throughput"] )
    except:
        print("Something went wrong when loading the throughput curve. Check file name!")
        quit()

    ## General Constants ---------------
    speedoflight = np.log10(29979245800) # speed of light in cm/2
    hplank = np.log10(6.626068)-27 # Plank constant in erg*s
    loghc = speedoflight + hplank # H*c in log(erg * cm)
    f_nu_AB = 48.59

    ## Sky Constants ---------------
    
    # Sky transparency
    # Not used since in space

    # Sky Background
    # lam is in micron
    # Bkg are in MJy/sr and need to be converted to photons/s/arcsec2/nm/m2
    try:
        skybackground = ascii.read(os.path.join(DIRS["bk_path"] , telstats["bk_filename"]) , names=["lam","lowBkg","midBkg","highBkg"] )
    except:
        print("Something went wrong when reading the sky background file. Check file name!")
        quit()

    bkg = skybackground["midBkg"].copy()
    bkg = bkg / 206265**2 # from MJy/sr to MJy/arcsec2
    bkg = bkg * 1e6 # from MJy/arcsec2 to Jy/arcsec2
    bkg = bkg * 1e-26 # from Jy/arcsec2 to W/m2/s/Hz/arcsec2
    bkg = bkg * 1e7 # from W/m2/s/Hz/arcsec2 to erg/m2/s/Hz/arcsec2
    bkg = bkg / (10**loghc / (skybackground["lam"]*1e-4)) # from erg/m2/s/Hz/arcsec2 to ph/m2/s/Hz/arcsec2
    bkg = bkg * 10**speedoflight / (skybackground["lam"]*1e-4)**2 # from ph/m2/s/Hz/arcsec2 to ph/m2/s/cm/arcsec2
    bkg = bkg * 1e-7 # from ph/m2/s/cm/arcsec2 to phm2/s/nm/arcsec2
    skybackground["Bkg"] = bkg.copy()


    ### ============= CHECK SINGLE LINE INPUT ======================
    if (userinput["lineF"] > 0) & (userinput["lineW"] > 0) & (userinput["specFile"] == "none"):
        print("Single Line Input!")
        SPECTYPE = "line"

        ## Put Everything in micron
        if userinput["InputInAngstroms"] == True:
            userinput["lineW"] = userinput["lineW"]/10000.0

        ## Put in observed frame
        line = userinput["lineW"] * (1+userinput["z"])

        ## Check to see if the band is correct
        if (line > np.nanmax(filter_trans["lam"])) | (line < np.nanmin(filter_trans["lam"])):
            print("This wavelength %g is not contained in the %s band" % (line,userinput["band"]))
            quit()

    ### ============= CHECK USER INPUT SPECTRUM ======================
    #elif ((userinput["lineF"] < 0) | (userinput["lineW"] < 0)) & (userinput["specFile"] != "none"):
    elif (userinput["specFile"] != "none") & (userinput["specFile"] != "flat"):  
        print("Loading User Spectrum!")
        SPECTYPE = "user"
        
        # tell the user if 'mag' is used to normalize spectrum
        if userinput["NormalizeUserSpec"] == True:
            if userinput["mag"] > 0:
                print("Using 'mag' to normalize the user spectrum")
            else:
                print("The magnitude (%g AB) for normalization looks weird. Check it!" % userinput["mag"])
                quit()
        else:
            if userinput["mag"] > 0:
                print("Warning: Your magnitude to normalize the spectrum looks reasonable but is not used because 'NormalizeUserSpec' is set to False.")
        
        # load
        try:
            user_spec = ascii.read(userinput["specFile"] , names=["lam","flux"])
        except:
            print("Something went wrong when loading the user spectrum. Check file name!")
            quit()

        
        
        # convert to microns if needed
        if userinput["InputInAngstroms"] == True:
            user_spec["lam"] = user_spec["lam"] / 10000.0
        
        # convert to observed
        user_spec["lam"] = user_spec["lam"] * (1+userinput["z"])
        user_spec["flux"] = user_spec["flux"] / (1+userinput["z"])

        # check if filter covers the spectrum
        sel_real = np.where(filter_trans["trans"] > 0.01)[0] # don't really know why we have choose 0.01 here
        filter_lam_sel = filter_trans["lam"][sel_real]
        if (np.nanmin(user_spec["lam"]) > np.nanmin(filter_lam_sel) ) | ( np.nanmax(user_spec["lam"]) < np.nanmax(filter_lam_sel)  ):
            print("The read-in spectrum from %s does not span the full wavelength coverage of the %s band or is not in the proper format. The correct format is observed-frame wavelength in microns or Angstroms and flux in erg/s/cm2 in two column format. Please also check the wavelength unit." % (userinput["specFile"],userinput["band"]))
            quit()

    ### ============= FLAT INPUT SPECTRUM ======================
    #elif ((userinput["lineF"] < 0) | (userinput["lineW"] < 0)) & (userinput["specFile"] == "none"):
    elif userinput["specFile"] == "flat":
        print("Assuming a flat spectrum in f_nu (using mag=%g AB to normalize spectrum)!" % userinput["mag"])
        SPECTYPE = "flat"

        if userinput["mag"] < 0:
            print("The magnitude (%g AB) for the normalization of the flat input spectrum looks weird. Please check!" % userinput["mag"])
            quit()


    else:
        print("Something is wrong with the input spectrum or line")
        quit()



    ### =========== FIGURE OUT THROUGHPUT AND RESOLUTION ================

    # First set things less than 0 to 0. Just in case
    sel_zero = np.where( throughput["throughput"] < 0)[0]
    throughput["throughput"][sel_zero] = 0

    # the total throughput is the instrument throughput
    # times the reflectance of the keck primary and secondary
    throughput["throughput"] = throughput["throughput"] * telstats["Mref"]**telstats["Nmirror"]

    # real FWHM resolution
    R = telstats["rt"] / userinput["slit_width"]

    # slit width in pixels along the dispersion direction
    #swp = userinput["slit_width"] / telstats["pix_disp"]

    # spectral coverage in microns
    #cov = telstats["tot_Npix"] * (telstats["disp"] / 10000.0)


    ### ========== CONVOLVE FILTER, THROUGHPUT, AND SKYBACKGROUND WITH RESOLUTION =======

    ## Convolve the filter with instrument resolution
    # output is dictionary, change the key names to make it consistent with IDL program
    # note that wave_grid is always the same if center wavelength is the same!
    filter_trans_convolved = degrade_resolution(in_wave=filter_trans["lam"],
                            in_flux=filter_trans["trans"],
                            center_wave=telstats["lambda"],
                            R=R,
                            disp=telstats["disp"],
                            tot_Npix=telstats["tot_Npix"]
                            )
    filter_trans_convolved["fltSpecObs"] = filter_trans_convolved.pop("flux")
    filter_trans_convolved["wave_grid"] = filter_trans_convolved.pop("lam")
    
    # the relevant portion of the spectrum
    band_index = np.where(filter_trans_convolved["fltSpecObs"] > 0.1)[0] # this is always the same for the degrade_resolution output as long as same center wavelength!
    filter_trans_convolved["fltSpecObs"] = filter_trans_convolved["fltSpecObs"][band_index]
    filter_trans_convolved["wave_grid"] = filter_trans_convolved["wave_grid"][band_index]

    # a thigher constraint for the S/N
    filt_index = np.where(filter_trans_convolved["fltSpecObs"] > 0.5)[0]


    # convolve throughput spectrum wiht the resolution
    # output is dictionary, change the key names to make it consistent with IDL program
    # note that wave_grid is always the same if center wavelength is the same!
    throughput_convolved = degrade_resolution(in_wave=throughput["tp_wave"],
                            in_flux=throughput["throughput"],
                            center_wave=telstats["lambda"],
                            R=R,
                            disp=telstats["disp"],
                            tot_Npix=telstats["tot_Npix"]
                            )
    throughput_convolved["tpSpecObs"] = throughput_convolved.pop("flux")
    throughput_convolved["wave_grid"] = throughput_convolved.pop("lam")
    throughput_convolved["tpSpecObs"] = throughput_convolved["tpSpecObs"][band_index]
    throughput_convolved["wave_grid"] = throughput_convolved["wave_grid"][band_index]

    # convolve the background spectrum with the resolution
    # background in phot/sec/arcsec^2/nm/m^2
    # output is dictionary, change the key names to make it consistent with IDL program
    # note that wave_grid is always the same if center wavelength is the same!
    skybackground_convolved = degrade_resolution(in_wave=skybackground["lam"],
                            in_flux=skybackground["Bkg"],
                            center_wave=telstats["lambda"],
                            R=R,
                            disp=telstats["disp"],
                            tot_Npix=telstats["tot_Npix"]
                            )
    skybackground_convolved["raw_bkSpecObs"] = skybackground_convolved.pop("flux")
    skybackground_convolved["wave_grid"] = skybackground_convolved.pop("lam")
    sel_zero = np.where( skybackground_convolved["raw_bkSpecObs"] < 0)[0]
    #sel_nonzero = np.where( (skybackground_convolved["raw_bkSpecObs"] > 0) & (skybackground_convolved["raw_bkSpecObs"] != -0) )[0]

    skybackground_convolved["raw_bkSpecObs"][sel_zero] = 0

    skybackground_convolved["raw_bkSpecObs"] = skybackground_convolved["raw_bkSpecObs"][band_index]
    skybackground_convolved["wave_grid"] = skybackground_convolved["wave_grid"][band_index]

    # this is the final wave grid to be used in the following
    wave_grid = skybackground_convolved["wave_grid"].copy()

    


    ### =========== USING A SPECIFIC LINE FLUX =================
    #if (userinput["lineF"] > 0) & (userinput["lineW"] > 0):
    if SPECTYPE == "line":

        ## Figure out the relevant wavelength range and other things --------

        # The Observed line wavelength in micron
        center = userinput["lineW"] * (1+userinput["z"])

        # resolution at the central wavelength
        res = center / R

        # the width of the spectral line before going through the spectrograph
        real_width = center * userinput["FWHM"] * 10**(-speedoflight + 5)

        # the line width in microns that should be observed
        width = np.sqrt(real_width**2 + res**2)

        
        ## Compute spectrum -------------

        # the location inside the FWHM of the line
        line_index = np.where( np.abs(wave_grid - center) <= 0.5*width)[0]

        # check if line is wide enough
        if len(line_index) == 0:
            print("ERROR: Line is too narrow.")
            quit()
        else:
            pass

        # The area used to calculate the S/N
        sn_index = line_index.copy()

        # now send the background spectrum through the telescope by 
        # multiplying the throughput, the
        # slit_width, the angular extent (theta), the area of the
        # telescope, and the pixel scale in nm
        # This gives phot/sec/pixel
        skybackground_convolved["bkSpecObs"] = skybackground_convolved["raw_bkSpecObs"] * throughput_convolved["tpSpecObs"] * userinput["slit_width"] * userinput["theta"] * (telstats["AT"] * 10**(-4.)) * (telstats["disp"]/10.0)

        # determine the average background within the FWHM of the line
        # This is no longer true
        # In photons per second per arcsec^2 per nm per m^2
        #  sel_tmp = np.where( np.abs(skybackground["lam"] - center) < 0.5*width )[0]
        #  averageBkg = np.nanmean(skybackground["Bkg"][sel_tmp])
        # This doesnt work anymore because the resolution of the skybackground lambda is too low.
        # Instead, I now interpolate
        averageBkg = np.interp(center,skybackground['lam'],skybackground["Bkg"])

        

        # what does this correspond to in AB mags/arcsec^2
        # go to erg/s/cm^2/Hz
        # 10^-4 for m^2 to cm^2
        # 10^3 for micron to nm
        # lam^2/c to covert from d(lam) to d(nu) (per Hz instead of per nm)
        # hc/lam to convert photons to ergs
        mag_back = -2.5 * ( np.log10(averageBkg*center) - 4 + 3 + hplank ) - f_nu_AB

        # the signal in electrons per second that will hit the telsecope
        # as it hits the atmosphere (ie need
        # to multiply by the throughput and
        # the atmospheric transparency
        signalATM = userinput["lineF"] * 10**(-18-loghc-4) * center * telstats["AT"]

        # the width of the line in sigma - not FWHM
        # in micron
        sigma = width / (2 * np.sqrt(2 * np.log(2)))

        # a spectrum version of the signal
        # phot per second per pixel (without atm or telescope)
        # ie total(signal_spec/signal) with
        # equal resolution of wave_grid / stat.disp in micron
        signal_spec = signalATM * ( 1 / ( np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ( wave_grid - center)**2 / sigma**2) * telstats["disp"] / 1e4

        # the spectrum of the signal as detected (in space!)
        sigSpecObs = signal_spec * throughput_convolved["tpSpecObs"] * 1 #tranSpecObs

        # the number of pixels in the spectral direction
        nPixSpec = (width * 10000.0) / telstats["disp"]

        # the spatial pixel scale
        nPixSpatial = userinput["theta"] / telstats["pix_scale"]

        # The number of pixels per FWHM observed
        Npix = nPixSpec * nPixSpatial


    else: ## =========== USING THE USER INPUT SPECTRUM OR FLAT SPECTRUM ==========
        
        ## We are calculating for a broad band flux

        # The observed line wavelength in micron (here the center of the band)
        center = telstats["lambda"]

        # resolution at the central wavelength in micron
        res = center / R

        # the area used to calculate the S/N
        sn_index = filt_index.copy()

        # background magnitude
        mag_back = -2.5 * ( np.log10( np.nanmean( skybackground_convolved["raw_bkSpecObs"][filt_index]) * center) - 4 + 3 + hplank) - f_nu_AB

        # now send the background spectrum through the telescope by 
        # multiplying the throughput, the
        # slit_width, the angular extent, the area of the
        # telescope, and the pixel scale in nm
        # this gives phot/sec/pixel
        skybackground_convolved["bkSpecObs"] = skybackground_convolved["raw_bkSpecObs"] * throughput_convolved["tpSpecObs"] * userinput["slit_width"] * userinput["theta"] * (telstats["AT"] * 10.**(-4)) * (telstats["disp"] / 10.0)

        #if userinput["specFile"] != "none":
        if SPECTYPE == "user": # for USER spectrum

            # convolve the user spectrum with the resolution
            # output is dictionary, change the key names to make it consistent with IDL program
            # note that wave_grid is always the same if center wavelength is the same!
            user_spec_convolved = degrade_resolution(in_wave=user_spec["lam"],
                                    in_flux=user_spec["flux"],
                                    center_wave=telstats["lambda"],
                                    R=R,
                                    disp=telstats["disp"],
                                    tot_Npix=telstats["tot_Npix"]
                                    )
            user_spec_convolved["userSig"] = user_spec_convolved.pop("flux")
            user_spec_convolved["user_wave_grid"] = user_spec_convolved.pop("lam")

            user_spec_convolved["userSig"] = user_spec_convolved["userSig"][band_index]
            user_spec_convolved["user_wave_grid"] = user_spec_convolved["user_wave_grid"][band_index]

            
            

            # multiply by the normalized filter transmission
            filt_shape = filter_trans_convolved["fltSpecObs"] / np.nanmax(filter_trans_convolved["fltSpecObs"])
            user_spec_convolved["userSig"] = user_spec_convolved["userSig"] * filt_shape

            # Check if User Spectrum needs to be normalized
            if userinput["NormalizeUserSpec"] == True:
                # make the total macth the broad band magnitude
                scale = 10.0**(-0.4 * (userinput["mag"] + f_nu_AB)) / np.nanmean(user_spec_convolved["userSig"])
                raw_fv_sig_spec = user_spec_convolved["userSig"] * scale
            
            else:
                raw_fv_sig_spec = user_spec_convolved["userSig"].copy()


            # convert to flux hitting the primary
            # in flux hitting the primary in phot/sec/micron 
            # (if the earth had no atmosphere)
            # phot/sec/micron = fnu * AT / lam / h
            signal_spec = raw_fv_sig_spec * 10.**(-1 * hplank) * telstats["AT"] / wave_grid

            

        elif SPECTYPE == "flat": ## for FLAT spectrum

            # flux hitting the primary in
            # phot/sec/micron (if the earth had no atmosphere)
            signal_spec = 10.0**(-0.4 * (userinput["mag"] + f_nu_AB) - hplank) * telstats["AT"] / wave_grid

        # multiply by the atmospheric transparency (in space!)
        signal_spec = signal_spec * 1 # tranSpecObs

        # now put it through the throughput of the telescope
        # phot/sec/micron
        sigSpecObs = signal_spec * throughput_convolved["tpSpecObs"]

        # now for phot/sec/pix multiply by micron/pix
        sigSpecObs = sigSpecObs * (telstats["disp"] / 10000.0)

        # number of pixels per resolution element in the spectral direction
        nPixSpec = (res*10000.0) / telstats["disp"]

        # the spatial pixel scale
        # we have at least one pixel in the cross dispersion direction
        nPixSpatial = np.nanmax( np.asarray([userinput["theta"] / telstats["pix_scale"],2]) )

        # The number of pixels per FWHM observed
        Npix = nPixSpec * nPixSpatial
        
    ## ================== FINALLY, COMPUTE =================

    ## Get Exposure time if a S/N is given by the user. ----------
    if userinput["SN"] > 0:

        print("S/N is given by user: compute exposure time from that (ignore exp. time user input)!")

        # differentiate between total exposure time 
        # and amount of time of individual exposures
        
        # figure out how long it takes

        # if calulating with a line flux, assume S/N over the line
        # other wise, S/N per spectral pixel
        if (userinput["lineF"] > 0) & (userinput["lineW"] > 0):
            qa = -nPixSpec * sigSpecObs**2 / userinput["SN"]**2
        else:
            qa= -sigSpecObs**2 / userinput["SN"]**2

        qb = dither * skybackground_convolved["bkSpecObs"] + dither * telstats["dark"] * nPixSpatial + sigSpecObs
        qc = dither * telstats["det_RN"]**2 / userinput["Nreads"] * nPixSpatial * userinput["nExp"]

        # note that if input is an emission line, some of the sigSpecObs=0 and
        # therefore some of the qa and qb are 0 as well. This generates a warning
        # in the division by qa below. Limit calculation already here to sn_index.
        if len(sn_index) == 0:
            print("ERROR: No signal detected when computing exposure time for given input S/N.")
            quit()
        timeSpec = (-qb[sn_index] - np.sqrt( qb[sn_index]**2 - 4 * qa[sn_index] * qc )) / (2 * qa[sn_index])
        
        # take median to get exposure time
        time = np.float( np.nanmedian( timeSpec) )

    else: ## Else, take the exposure time given by the user -----
        time = userinput["time"]


    ## Compute the S/N --------

    # noise contributions
    # - poisson of background
    # - poisson of dark current
    # - poisson of the signal
    # - read noise
    # the noise per slit length in the spatial direction 
    # and per pixel in the spectral direction
    #
    # the noise spectrum: 
    # Poisson of the dark
    # current, signal, and background + the read noise"
    noiseSpecObs = np.sqrt( sigSpecObs * time + dither * ( (skybackground_convolved["bkSpecObs"] + telstats["dark"] * nPixSpatial) * time + telstats["det_RN"]**2 / userinput["Nreads"] * nPixSpatial * userinput["nExp"]))

    signalSpecObs =  sigSpecObs * time

    snSpecObs = signalSpecObs / noiseSpecObs

    stn = np.nanmean(np.sqrt(nPixSpec) * snSpecObs[sn_index])

    # the electron per pixel spectrum
    eppSpec = noiseSpecObs**2 / nPixSpatial


    ## ============= VALUES TO BE PRINTED ===========

    # the mean instrument+telescope throughput in
    # the same band pass
    tp = np.nanmean( throughput_convolved["tpSpecObs"][sn_index])

    # maximum electron per pixel
    max_epp = np.max( eppSpec[sn_index] / userinput["nExp"] )

    if (userinput["lineF"] > 0) & (userinput["lineW"] > 0): # if calulating a line flux, S/N per FWHM ie S/N in the line

        # over the line (per FWHM)
        stn = np.nanmean(np.sqrt(nPixSpec) * snSpecObs[sn_index])

        # signal in e/FWHM
        signal = np.nanmean(sigSpecObs[sn_index]) * nPixSpec * time

        # sky background in e/sec/FWHM
        background = np.nanmean( skybackground_convolved["bkSpecObs"][sn_index]) * nPixSpec * time
    
        # Read noise for multiple reads, electrons per FWHM
        RN = telstats["det_RN"] / np.sqrt(userinput["Nreads"]) * np.sqrt(Npix) * np.sqrt(userinput["nExp"])

        # noise per FWHM
        noise = np.nanmean( noiseSpecObs[sn_index]) * np.sqrt(nPixSpec)

        # e- 
        dark = telstats["dark"] * Npix * time
    
    else: # we are computing S/N per pixel for a continuum source

        # per spectral pixel
        stn = np.nanmedian( snSpecObs[sn_index] )

        # signal in e/(spectral pixel)
        signal = np.nanmedian( sigSpecObs[sn_index]) * time

        # sky background in e/(spectral pixel)
        background = np.nanmedian( skybackground_convolved["bkSpecObs"][sn_index]) * time

        # Read noise for multiple reads, electrons per spectral pixel
        RN = telstats["det_RN"] / np.sqrt(userinput["Nreads"]) * np.sqrt(nPixSpatial) * np.sqrt(userinput["nExp"])

        # noise per spectral pixel
        noise = np.nanmedian(noiseSpecObs[sn_index])

        # e- per spectral pixel
        dark = telstats["dark"] * nPixSpatial * time



    ## =============== CREATE OUTPUT =============
    
    ## Summary dictionary -------------
    summary_struct = dict()
    summary_struct["quant"] = ['Wavelength', 'Resolution','Dispersion', 'Throughput', 'Signal', 'Sky Background', 
                       'Sky brightness', 'Dark Current', 'Read Noise', 'Total Noise','S/N', 
                       'Total Exposure Time', 'Max e- per pixel']
    
    if (userinput["lineF"] > 0) & (userinput["lineW"] > 0):
        summary_struct["unit"] = ['micron','FWHM in angstrom', 'angstrom/pixel', '',  'electrons per FWHM',
         'electrons per FWHM', 'AB mag per sq. arcsec', 'electrons per FWHM', 
         'electrons per FWHM', 
         'electrons per FWHM',
         'per observed FWHM', 'seconds', 'electrons per pixel per exp']
    else:
        summary_struct["unit"] = ['micron','angstrom', 'angstrom/pixel', '',  'electrons per spectral pixel',
         'electrons per spectral pixel', 'AB mag per sq. arcsec', 'electrons per spectral pixel', 
         'electrons per spectral pixel', 'electrons per spectral pixel',
         'per spectral pixel', 'seconds', 'electrons per pixel']
    

    if max_epp >= 1e10:
        max_epp_string = "> 1e10"
    else:
        max_epp_string = max_epp

    if max_epp > telstats["sat_limit"]:
        print("Detector Saturated!")
    elif (max_epp > telstats["five_per_limit"]) & (max_epp < telstats["sat_limit"]):
        print("Detector in >5 percent unlinear regime")
    elif (max_epp > telstats["one_per_limit"]) & (max_epp < telstats["five_per_limit"]):
        print("Detector in 1 - 5 percent nonlinear regime")
    else:
        pass

    
    summary_struct["value"] = [round(center,4),
                    round(res * 1e4,1),
                    round(telstats["disp"],2),
                    tp,
                    signal,
                    background,
                    mag_back,
                    dark,
                    RN,
                    noise,
                    stn,
                    time,
                    max_epp_string
                    ]

    ## Actual output containing the spectrum --------------
    spec_struct = dict()
    
    spec_struct["wave"] = wave_grid
    spec_struct["center"] = center
    spec_struct["plot_index"] = sn_index
    spec_struct["filt_index"] = filt_index
    spec_struct["tp"] = throughput_convolved["tpSpecObs"]
    spec_struct["filt"] = filter_trans_convolved["fltSpecObs"]
    spec_struct["bk"] = skybackground_convolved["bkSpecObs"]
    spec_struct["sig"] = signal_spec # phot/sec/micron
    spec_struct["signal"] = signalSpecObs # phot/sec/micron
    spec_struct["noise"] = noiseSpecObs
    spec_struct["sn"] = snSpecObs
    spec_struct["lineF"] = userinput["lineF"]
    spec_struct["time"] = time

    output = {"summary_struct":summary_struct,
                "spec_struct":spec_struct}

    return(output)


## Function to create summary plots from the output of atlas_etc
def etc_plot(spec_struct, filename):
    '''
    Function to create summary plot from the output of atlas_etc(). \\
    USAGE: etc_plots(spec_struct , filename) \\
        where \\
        "spec_struct" is the "spec_struct" output of the atlas_etc() function \\
        "filename" is the file name (and path) where the PDF plots should be saved. \\
        Note that if "filename" is set to "none", then the plots are just displayed but not saved. 
    '''

    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ## Plot observed User spectrum and noises
    ax1.plot(spec_struct["wave"] , spec_struct["signal"] , label="Science")
    ax1.plot(spec_struct["wave"] , spec_struct["bk"] , label="Sky Background")
    ax1.plot(spec_struct["wave"] , spec_struct["noise"] , label="Noise")
    ax1.legend(bbox_to_anchor=(0.9, 1.1) , ncol=3)
    #ax1.set_title("Observed Signal")
    ax1.set_xlabel(r"Wavelength ($\mu$m)")
    ax1.set_ylabel(r"Photons per Pixel")

    ## Plot transmission in same plot (ax1)
    ax11 = ax1.twinx()
    ax11.plot(spec_struct["wave"],spec_struct["tp"] , dashes=(2,2), linewidth=1 , color="black" , label="Spectral Throughput")
    ax11.plot(spec_struct["wave"],spec_struct["filt"] , dashes=(5,5), linewidth=1, color="lightgray" , label="Filter Throughput")
    ax11.set_ylabel("Transmission")
    ax11.legend(bbox_to_anchor=(0.9, 1.18) , ncol=2)

    ## Plot S/N
    ax2.plot(spec_struct["wave"] , spec_struct["sn"] , label="S/N per Pixel")
    ax2.legend(bbox_to_anchor=(1, 1.1) , ncol=1)
    ax2.set_xlabel(r"Wavelength ($\mu$m)")
    ax2.set_ylabel(r"S/N per Pixel")



    if filename == "none":
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    #plt.plot(spec_struct["wave"],spec_struct["sn"] , label="S/N")
    ##plt.plot(spec_struct["wave"],spec_struct["signal"] , label="Signal")
    ##plt.plot(spec_struct["wave"],spec_struct["noise"] , label="Noise")
    #plt.legend(loc="upper right")
    #plt.xlim(1.63,1.65)
    #plt.show()




#################
