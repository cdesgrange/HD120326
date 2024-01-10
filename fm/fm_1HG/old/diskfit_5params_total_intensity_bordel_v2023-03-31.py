# pylint: disable=C0103
####### This is the MCMC fitting code for fitting a disk #######
import os, sys, time

import socket

import distutils.dir_util
import warnings

from multiprocessing import cpu_count
from multiprocessing import Pool

import contextlib

from datetime import datetime

import math as mt
import numpy as np

import scipy.optimize as op

import astropy.io.fits as fits
from astropy.convolution import (convolve_fft, convolve, Gaussian2DKernel)

import yaml

import pyklip.instruments.Instrument as Instrument

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm

from emcee import EnsembleSampler
#from emcee import backends

# import make_gpi_psf_for_disks as gpidiskpsf

import vip_hci as vip

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk
from vip_hci.preproc.recentering import frame_shift

import astro_unit_conversion as convert

from derive_noise_map import *

import pyregion

os.environ["OMP_NUM_THREADS"] = "1"

#from functions_diskfit_mcmc import *


#######################################################
########### 5 params: a, PA, inc, g, scaling ##########
#######################################################
def call_gen_disk_5params(theta):
    """ call the disk model from a set of parameters. 1g SPF
        use global variables DIMENSION, PIXSCALE_INS and DISTANCE_STAR

    Args:
        theta: list of parameters of the MCMC
        here 5 params: a, PA, inc, g, scaling

    Returns:
        a 2d model
    """
    r1 = theta[0]
    pa = theta[1]
    inc = theta[2] #np.degrees(np.arccos(theta[2]))
    g1 = theta[3]
    scaling = theta[4]
     
    # Fixed parameters
    ain = 10#12#10
    aout = -2#-1.9 #-5 
   
    argperi = 0 
    eccentricity = 0
    ksi0 = 0.1756 


    # Generate the model
    model = ScatteredLightDisk(nx=DIMENSION,
                               ny=DIMENSION,
                               distance=DISTANCE_STAR,
                               itilt=-inc,
                               omega=argperi,
                               pxInArcsec=PIXSCALE_INS,
                               pa=pa,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': ain,
                                   'aout': aout,
                                   'a': r1,
                                   'e': eccentricity,
                                   'ksi0': ksi0,
                                   'gamma': 2.,
                                   'beta': 1.
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': g1,
                                   'polar': True
                               })
    
    return model.compute_scattered_light() * scaling


########################################################
def chisquare_5params(theta):
    """ measure the Chisquare of the parameter set.
        create disk
        convolve by the PSF (psf is global)
        do the forward modeling (diskFM obj is global)
        nan out when it is out of the zone (zone mask is global)
        subctract from data and divide by noise (data and noise are global)

    Args:
        theta: list of parameters of the MCMC

    Returns:
        Chisquare
    """
    # Generate disk model
    model = call_gen_disk_5params(theta)

    # Rotate the disk model for different angles
    # and convolve it by the PSF
    modelconvolved = vip.fm.cube_inject_fakedisk(model, PA_RANGE, psf=PSF)

    # Compute the residuals and the chisquare

    # Remove the disk from the observation
    CUBE_DIFF = (REDUCED_DATA - modelconvolved)

    # Reduce the observation
    #im_pca, pcs, reconstr_cube, res_cube, res_cube_derot = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_RANGE, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=False)
    im_pca = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_RANGE, ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

    res = im_pca / NOISE
    Chisquare = np.nansum(res * res)
    
    print('For theta =', theta, '\n-> Chisquare = {:.3e} i.e. {:.0f}'.format(Chisquare, Chisquare))

    return Chisquare


########################################################
def logl_5params(theta):
    """ measure the log of the likelihood of the parameter set.
        create disk
        convolve by the PSF (psf is global)
        do the forward modeling (diskFM obj is global)
        nan out when it is out of the zone (zone mask is global)
        subctract from data and divide by noise (data and noise are global)

    Args:
        theta: list of parameters of the MCMC

    Returns:
        Chisquare
    """
    ll = -0.5 * chisquare_5params(theta)
    return ll



########################################################
def from_param_to_theta_init_5params(params_mcmc_yaml):
    """ create a initial set of MCMCparameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    rad_init = params_mcmc_yaml['rad_init']
    pa_init = params_mcmc_yaml['pa_init']
    inc_init = params_mcmc_yaml['inc_init']
    g1_init = params_mcmc_yaml['g1_init']
    scaling = params_mcmc_yaml['scaling']

    #theta_init = (rad_init, pa_init, np.cos(np.radians(inc_init)), g1_init, scaling)
    theta_init = (rad_init, pa_init, inc_init, g1_init, scaling)

    for parami in theta_init:
        if parami == 0:
            raise ValueError("""Do not initialize one of your parameters
            at exactly 0.0, it messes up the small ball at MCMC initialization"""
                             )
    return theta_init




########################################################
def logp(theta):
    """ measure the log of the priors of the parameter set.

    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors
    """

    r1 = mt.exp(theta[0])
    ain = theta[1]
    aout = theta[2]
    inc = np.degrees(np.arccos(theta[3]))
    pa = theta[4]

    argperi = theta[5]
    eccentricity = theta[6]
    ksi0 = theta[7]

    g1 = theta[8]
    g2 = theta[9]
    alpha = theta[10]

    norm = mt.exp(theta[11])

    if (r1 < 20 or r1 > 130):  #Can't be bigger than 200 AU
        return -np.inf

    if (ain < 1 or aout > 30):
        return -np.inf

    if (aout < -30 or aout > -1):
        return -np.inf

    if (inc < 0 or inc > 90):
        return -np.inf

    if (pa < 0 or pa > 180):
        return -np.inf

    if argperi != None:
        if (argperi < 0 or argperi > 180):
            return -np.inf

    if eccentricity != None:
        if (eccentricity < 0 or eccentricity > 1):
            return -np.inf

    if (ksi0 < 0.1 or ksi0 > 10):  #The aspect ratio
        return -np.inf

    if (g1 < 0.05 or g1 > 0.9999):
        return -np.inf

    if g2 != None:
        if (g2 < -0.9999 or g2 > -0.05):
            return -np.inf

    if (alpha < 0.01 or alpha > 0.9999):
        return -np.inf

    if (norm < 0.5 or norm > 50000):
        return -np.inf
    # otherwise ...

    return 0.0


########################################################
def lnpb(theta):
    """ sum the logs of the priors (return of the logp function)
        and of the likelihood (return of the logl function)


    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors + log of likelyhood
    """
    # from datetime import datetime
    # starttime=datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    # print("Running time model + FM: ", datetime.now()-starttime)

    return lp + ll


########################################################
def make_disk_mask(dim,
                   estimPA,
                   estiminclin,
                   estimminr,
                   estimmaxr,
                   xcen=140.,
                   ycen=140.):
    """ make a zeros mask for a disk


    Args:
        dim: pixel, dimension of the square mask
        estimPA: degree, estimation of the PA
        estiminclin: degree, estimation of the inclination
        estimminr: pixel, inner radius of the mask
        estimmaxr: pixel, outer radius of the mask
        xcen: pixel, center of the mask
        ycen: pixel, center of the mask

    Returns:
        a [dim,dim] array where the mask is at 0 and the rest at 1
    """

    PA_rad = (90 + estimPA) * np.pi / 180.
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin * np.pi / 180.)
    rho2dellip = np.sqrt(x**2 + y**2)

    mask_object_astro_zeros = np.ones((dim, dim))
    mask_object_astro_zeros[np.where((rho2dellip > estimminr)
                                     & (rho2dellip < estimmaxr))] = 0.

    return mask_object_astro_zeros


########################################################
def make_noise_map_no_mask(reduced_data, xcen=140., ycen=140., delta_raddii=3):
    """ create a noise map from a image using concentring rings
        and measuring the standard deviation on them

    Args:
        reduced_data: [dim dim] array containing the reduced data
        xcen: pixel, center of the mask
        ycen: pixel, center of the mask
        delta_raddii: pixel, widht of the small concentric rings

    Returns:
        a [dim,dim] array where each concentric rings is at a constant value
            of the standard deviation of the reduced_data
    """

    dim = reduced_data.shape[1]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0, int(np.floor(xcen / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(reduced_data[wh_rings])
    return noise_map





########################################################
def initialize_walkers_backend(params_mcmc_yaml):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        if new_backend ==1 then [intial position of the walkers, a clean BACKEND]
        if new_backend ==0 then [None, the loaded BACKEND]
    """

    # if new_backend = 0, reset the backend, if not restart the chains.
    # Be careful if you change the parameters or walkers #, you have to put new_backend = 1
    new_backend = params_mcmc_yaml['NEW_BACKEND']

    nwalkers = params_mcmc_yaml['NWALKERS']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')

    theta_init = from_param_to_theta_init(params_mcmc_yaml)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc.h5")
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Don't worry, the walkers quickly branch out and explore the
    # rest of the space.
    if new_backend == 1:
        init_ball0 = np.random.uniform(theta_init[0] * 0.999,
                                       theta_init[0] * 1.001,
                                       size=(nwalkers))
        init_ball1 = np.random.uniform(theta_init[1] * 0.99,
                                       theta_init[1] * 1.01,
                                       size=(nwalkers))
        init_ball2 = np.random.uniform(theta_init[2] * 0.99,
                                       theta_init[2] * 1.01,
                                       size=(nwalkers))
        init_ball3 = np.random.uniform(theta_init[3] * 0.99,
                                       theta_init[3] * 1.01,
                                       size=(nwalkers))
        init_ball4 = np.random.uniform(theta_init[4] * 0.99,
                                       theta_init[4] * 1.01,
                                       size=(nwalkers))
        init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                       theta_init[5] * 1.01,
                                       size=(nwalkers))
        init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                       theta_init[6] * 1.01,
                                       size=(nwalkers))
        init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                       theta_init[7] * 1.01,
                                       size=(nwalkers))
        init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                       theta_init[8] * 1.01,
                                       size=(nwalkers))
        init_ball9 = np.random.uniform(theta_init[9] * 0.99,
                                       theta_init[9] * 1.01,
                                       size=(nwalkers))
        init_ball10 = np.random.uniform(theta_init[10] * 0.99,
                                        theta_init[10] * 1.01,
                                        size=(nwalkers))
        init_ball11 = np.random.uniform(theta_init[11] * 0.99,
                                        theta_init[11] * 1.01,
                                        size=(nwalkers))

        p0 = np.dstack((init_ball0, init_ball1, init_ball2, init_ball3,
                        init_ball4, init_ball5, init_ball6, init_ball7,
                        init_ball8, init_ball9, init_ball10, init_ball11))

        backend_ini.reset(nwalkers, n_dim_mcmc)
        return p0[0], backend_ini

    return None, backend_ini


########################################################
def from_param_to_theta_init(params_mcmc_yaml):
    """ create a initial set of MCMC parameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    rad_init = params_mcmc_yaml['rad_init']
    ain_init = params_mcmc_yaml['ain_init']
    aout_init = params_mcmc_yaml['aout_init']
    inc_init = params_mcmc_yaml['inc_init']
    pa_init = params_mcmc_yaml['pa_init']
    argperi_init = params_mcmc_yaml['argperi_init']
    ecc_init = params_mcmc_yaml['ecc_init']
    ksi0_init = params_mcmc_yaml['ksi0_init']
    g1_init = params_mcmc_yaml['g1_init']
    g2_init = params_mcmc_yaml['g2_init']
    alpha_init = params_mcmc_yaml['alpha_init']
    N_init = params_mcmc_yaml['N_init']

    if eval(g2_init) == None: g2_init = None
    #print(g2_init, type(g2_init))

    theta_init = (np.log(rad_init), ain_init, aout_init,
                  np.cos(np.radians(inc_init)), pa_init, argperi_init,
                  ecc_init, ksi0_init, g1_init, g2_init, alpha_init,
                  np.log(N_init))

    for parami in theta_init:
        if parami == 0:
            raise ValueError("""Do not initialize one of your parameters
            at exactly 0.0, it messes up the small ball at MCMC initialization"""
                             )

    return theta_init



if __name__ == '__main__':
    display = 1
    L = time.localtime()
    date = "{}-{}-{}-{}h{}min{}s".format(L[0],L[1],L[2],L[3],L[4],L[5],L[6])
        
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)
    
    # test on which machine I am
    if display: print(socket.gethostname())
    if socket.gethostname() == 'MacBook-Pro-de-sysipag.local':
        #datadir = '/Users/desgranc/Documents/work/projects/HD120326/data/'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        #datadir = '/Users/desgranc/Documents/work/projects/HD120326/data/'
        progress = False


    ## INITIALIZATION ##
    print('\n=== Initialization ===')
    if len(sys.argv) == 1:
        #str_yalm = 'SPHERE_Hband_MCMC.yaml'
        str_yalm = 'SPHERE_Hband_AMOEBA_total_intensity_2019-07-09_5params.yaml'
    else:
        str_yalm = sys.argv[1]
        
    if display: print(str_yalm)

    # Open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files', str_yalm)
    if display: print('\nThe configuration file .yaml is located at:\n', yaml_path_file)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
        TYPE_OBS = params_mcmc_yaml['TYPE_OBS']

    # Initialize variables
    DATADIR = params_mcmc_yaml['DATADIR']
    PREPROCESS = params_mcmc_yaml['PREPROCESS']
    POSTPROCESS = params_mcmc_yaml['POSTPROCESS']
    if TYPE_OBS == 'polar':
        FN_PSF_LEFT = params_mcmc_yaml['FN_PSF_LEFT']
        FN_PSF_RIGHT = params_mcmc_yaml['FN_PSF_RIGHT']
        FN_SCIENCE = params_mcmc_yaml['FN_SCIENCE']
        FN_NOISE = params_mcmc_yaml['FN_NOISE']
    else:
        FN_PSF = PREPROCESS + params_mcmc_yaml['FN_PSF']
        FN_PA  = PREPROCESS + params_mcmc_yaml['FN_PA']
        FN_SCIENCE = PREPROCESS + params_mcmc_yaml['FN_SCIENCE']
        FN_NOISE   = PREPROCESS + params_mcmc_yaml['FN_NOISE']
        METH_NOISE = params_mcmc_yaml['METH_NOISE']
        IBAND = params_mcmc_yaml['IBAND']
        NMODES = params_mcmc_yaml['NMODES']
        
    FN_MASK = params_mcmc_yaml['FN_MASK']

    # Saving directory
    SAVINGDIR = params_mcmc_yaml['SAVINGDIR'] + date
    os.makedirs(SAVINGDIR,exist_ok=True)

    if display:
        print('\nThe path to access data is:\n', DATADIR)
        print('The filename for the reduced image is:\n', FN_SCIENCE)
        print('The filename for the mask is:\n', FN_MASK)
        try: print('The filename for the PSF is:\n', FN_PSF)
        except:
            print('The filename for the PSF (left) is:\n', FN_PSF_LEFT)
            print('The filename for the PSF (right) is:\n', FN_PSF_RIGHT)

            
    ## LOAD VARIABLES AND MAKE THEM GLOBAL ##
    
    # Load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    if display:
        print('\n-Params:')
        print('The star is located at',  DISTANCE_STAR, 'pc.')
        print('The pixel scale is', PIXSCALE_INS, 'as/pix.')

    # Load science and processing parameters
    OWA = params_mcmc_yaml['OWA'] # initially
    cropping =  params_mcmc_yaml['CROPPING_SCIENCE']
    crop_noise =  params_mcmc_yaml['CROPPING_NOISE']
    crop_psf =  params_mcmc_yaml['CROPPING_PSF']
    DIMENSION = int(OWA - cropping*2) # after cropping
    MASK_RAD =  params_mcmc_yaml['MASK_RAD']
   
    noise_multiplication_factor = params_mcmc_yaml['NOISE_MULTIPLICATION_FACTOR']
    norm2contrast =  params_mcmc_yaml['SCIENCE_NORM']

    
    if display: print('The normalization factor used is:', norm2contrast)

    # Load PSF and make it global
    if TYPE_OBS  == 'polar':
        PSF_LEFT = fits.getdata(os.path.join(DATADIR, FN_PSF_LEFT))
        PSF_RIGHT = fits.getdata(os.path.join(DATADIR, FN_PSF_RIGHT))
        PSF = np.concatenate((PSF_LEFT, PSF_RIGHT))
        PSF = np.nansum(PSF, axis=0)
        PSF = frame_shift(PSF,0.5,0.5)
        PSF = PSF[crop_psf:-crop_psf+1,crop_psf:-crop_psf+1]
        
    else:
        PSF = fits.getdata(os.path.join(DATADIR, FN_PSF))
        PSF = np.nansum(PSF, axis=1)[IBAND]
    total_flux_psf = np.nansum(PSF)
    print('The total flux of the PSF image is:',total_flux_psf)
    PSF = PSF/total_flux_psf
    
    # Load parallactic angles
    PA_RANGE = -fits.getdata(os.path.join(DATADIR,FN_PA))

    # Load science cube
    if TYPE_OBS  == 'polar':
        REDUCED_DATA = fits.getdata(os.path.join(DATADIR, FN_SCIENCE))
    else:  REDUCED_DATA = fits.getdata(os.path.join(DATADIR, FN_SCIENCE))[IBAND]
    

    if display:
        print('\n-Initially')
        print('The size of the reduced image is:', np.shape(REDUCED_DATA))
        print('The size of the PSF is:', np.shape(PSF))
        print('The field of view is:', OWA)
        if TYPE_OBS != 'polar': print('The array size of the parallactic angles is:', np.shape(PA_RANGE))

    if TYPE_OBS  == 'polar':
        REDUCED_DATA = REDUCED_DATA[cropping:-cropping,cropping:-cropping]
    else :
        REDUCED_DATA = REDUCED_DATA[:,cropping:-cropping,cropping:-cropping]

    if METH_NOISE == 'compute_it':
        print('The noise map is derived by applying PCA on opposite parallactic angles.')
        NOISE_ALMOST =  vip.psfsub.pca_fullfr.pca(REDUCED_DATA, -PA_RANGE,
                                           ncomp=NMODES, mask_center_px=MASK_RAD,
                                           imlib='opencv', full_output=False)

        NOISE = compute_limdet_map_ann(NOISE_ALMOST, dr=2, alpha=1, center='center', even_or_odd='even')

        NOISE[NOISE==0] = np.nan #1e3
        print('The size of the noise map is:', np.shape(NOISE))

    else: print('The size of the noise map is:', np.shape(NOISE))

    if crop_noise != 0:
        NOISE = NOISE[crop_noise:-crop_noise,crop_noise:-crop_noise]
   
    if display:
        print('\n-After cropping:')
        print('The size of the reduced image is:', np.shape(REDUCED_DATA))
        print('The size of the noise map is:', np.shape(NOISE))
        print('The size of  the PSF is:', np.shape(PSF))
        print('The dimension of the image is:', DIMENSION)


    if 1:
        print('We reduced the data by applying PCA on opposite parallactic angles.')
        RED_DATA =  vip.psfsub.pca_fullfr.pca(REDUCED_DATA, PA_RANGE,
                                           ncomp=NMODES, mask_center_px=MASK_RAD,
                                           imlib='opencv', full_output=False)
        print('The size of the red data map is:', np.shape(RED_DATA))
        RED_DATA[cropping:-cropping,cropping:-cropping]
        print('The size of the red data map is now:', np.shape(RED_DATA))

    # Save NOISE MAP
    inputs_resultdir = SAVINGDIR + '/inputs'
    os.makedirs(inputs_resultdir,exist_ok=True)
    fits.writeto(os.path.join(inputs_resultdir,'NOISE.fits'),
                     NOISE, overwrite=True)
    
    # Define or Load the MASK within the residuals will be minimized
    if TYPE_OBS  == 'polar':
        mask2minimize = fits.getdata(FN_MASK)
        mask2minimize = mask2minimize[cropping:-cropping,cropping:-cropping]
    else:
        dum, header = fits.getdata(os.path.join(inputs_resultdir,'NOISE.fits'),header=True)
        file_2_ellipses = FN_MASK
        print('The mask loaded is:', FN_MASK)
        with open (file_2_ellipses, "r") as myfile:
            reg_2_ellipses=myfile.read() #.replace('\n', '')
        region_2_ellipses = pyregion.parse(reg_2_ellipses)
        myfilter = region_2_ellipses.get_filter(header=header)
        mask2minimize = (myfilter[1] & ~myfilter[0]).mask((NOISE.shape[0],NOISE.shape[1]))
        # NOISE map same spatial size than SCIENCE image
    
    if display: print('The size of the mask is:', np.shape(mask2minimize))
    
    # We multiply the reduced_data by the mask2minimize to avoid having
    # to pass it as a global variable
    mask2minimize = np.where(mask2minimize == False, 0, 1)
    REDUCED_DATA_MASK = np.copy(REDUCED_DATA) * mask2minimize

    params_to_fit = params_mcmc_yaml['params_to_fit']


    # Save files used for forward modelling, MCMC or AMOEBA
    print('\nSave input files at:\n', inputs_resultdir)

    if TYPE_OBS == 'polar':
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image.fits'),
                     REDUCED_DATA, overwrite=True)
    
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_masked.fits'),
                     REDUCED_DATA_MASK, overwrite=True)

    else :
        fits.writeto(os.path.join(inputs_resultdir,'cube_science.fits'),
                     REDUCED_DATA, overwrite=True)
        
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_opp_pa.fits'),
                     NOISE_ALMOST, overwrite=True)
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image.fits'),
                     RED_DATA, overwrite=True)
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_mask.fits'),
                     RED_DATA*mask2minimize, overwrite=True)
  
    fits.writeto(os.path.join(inputs_resultdir,'PSF.fits'),
                     PSF, overwrite=True)

    fits.writeto(os.path.join(inputs_resultdir,'mask2minimize.fits'),
                     mask2minimize, overwrite=True)
  

    exploration_algo = params_mcmc_yaml['exploration_algo']

    ## TESTS ## 
    if 1:
        print('\n\n=== Make final test before running the MCMC / AMOEBA exploration algorithm ===')
        firstguess_resultdir = SAVINGDIR + '/first_guess'
        os.makedirs(firstguess_resultdir,exist_ok=True)
        # What is the chisquare if the disk model is empty?
        print('\n- No disk')
        startTime = datetime.now()
        theta_init = from_param_to_theta_init_5params(params_mcmc_yaml)
        theta_init_no_disk = np.copy(theta_init)
        theta_init_no_disk[-1] = 0
        print('Check disk model parameters without flux (scaling_flux should be set to 0):\n', theta_init_no_disk)
        
        if exploration_algo == "MCMC":
             lnpb_model = lnpb(theta_init_no_disk)
             print("Test likelihood on initial model:", lnpb_model)
             
        elif exploration_algo == "AMOEBA":
            chisquare_init_nodisk = chisquare_5params(theta_init_no_disk)
            #print("Test chisquare on initial model: {:.0f}".format(chisquare_init_nodisk))

        print("Time for a single model: ", datetime.now() - startTime)

        
        # What is the chisquare / likelihood of the initial model?
  
        print('\n- Initial model')
        startTime = datetime.now()
        #print('Parameter starting point:', theta_init)
        
        if exploration_algo == "MCMC":
             lnpb_model = lnpb(theta_init)
             print("Test likelihood on initial model:", lnpb_model)
             suffix = '_{:.0f}'.format(lnpb_model)
             
        elif exploration_algo == "AMOEBA":
            chisquare_init = chisquare_5params(theta_init)
            #print("Test chisquare on initial model: {:.0f}".format(chisquare_init))
            suffix = '_{:.0f}.'.format(chisquare_init)
            print("chisquare (no-disk) - chisquare (first-guess): {:.0f}".format((chisquare_init_nodisk-chisquare_init)))

        if TYPE_OBS == 'polar':
            model = call_gen_disk_5params(theta_init)
            modelconvolved = convolve_fft(model, PSF, boundary='wrap')
            best_residuals = (REDUCED_DATA - modelconvolved)
            best_residuals_snr = best_residuals / NOISE

        else:
            # Generate disk model
            model = call_gen_disk_5params(theta_init)

            # Rotate the disk model for different angles
            # and convolve it by the PSF
            modelconvolved = vip.fm.cube_inject_fakedisk(model, PA_RANGE, psf=PSF)
            modelconvolved0 = convolve_fft(model, PSF, boundary='wrap')
            

            # Remove the disk from the observation
            CUBE_DIFF = REDUCED_DATA - modelconvolved

            # Reduce the observation
            best_residuals = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_RANGE, ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

            # Compute the residuals and the chisquare
            best_residuals_snr = best_residuals / NOISE
            Chisquare = np.nansum(best_residuals_snr * best_residuals_snr)

            fits.writeto(os.path.join(firstguess_resultdir,'first_guess_residuals_cube.fits'),
                     CUBE_DIFF, overwrite=True)
            
            
        print('Save first-guess files')
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_disk_model{}.fits'.format(suffix)),
                     model, overwrite=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_disk_model_convolved_cube.fits'),
                     modelconvolved, overwrite=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_disk_model_convolved.fits'),
                     modelconvolved0, overwrite=True)
        
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_residuals.fits'),
                     best_residuals, overwrite=True)
        
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_residuals_snr.fits'),
            best_residuals_snr, overwrite=True)


    
        print("Time for a single model: ", datetime.now() - startTime)



        
    if exploration_algo == "MCMC":
        print('\n\n=== Initialization MCMC ===')
        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(params_mcmc_yaml)

        # load the Parameters necessary to launch the MCMC
        NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
        N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
        N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']  #Number of MCMC dimension

        # last chance to remove some global variable to be as light as possible
        # in the MCMC
        del params_mcmc_yaml

        #Let's start the MCMC
        startTime = datetime.now()
        print("Start MCMC")
        with contextlib.closing(Pool()) as pool:
            # Set up the Sampler. I purposefully passed the variables (KL modes,
            # reduced data, masks) in global variables to save time as advised in
            # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
            sampler = EnsembleSampler(NWALKERS,
                                      N_DIM_MCMC,
                                      lnpb,
                                      pool=pool,
                                      backend=BACKEND)

            sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=progress)

        print("\n time for {0} iterations with {1} walkers and {2} cpus: {3}".
              format(N_ITER_MCMC, NWALKERS, cpu_count(),
                     datetime.now() - startTime))

    elif exploration_algo == "AMOEBA":
        print('\n=== Start AMOEBA optimization ===')
        startTime = datetime.now()
        theta_init = from_param_to_theta_init_5params(params_mcmc_yaml)
        bounds = [(10,100), (0,180), (0, 89), (0,1), (0,None)] #params_mcmc_yaml['bounds']
        
        print('The parameters to fit are:', params_to_fit)
        print('(!) The initial guess parameters are:', theta_init)
        print('The bounds are:', bounds)

        print(bounds, type(bounds))
        a = zip(*bounds)
        print(a)
        result_optim = op.minimize(chisquare_5params,
                                   theta_init,
                                   method='Nelder-Mead',
                                   #constraints=constr
                                   bounds=bounds,
                                   tol=10
                                   )
        print('Time for the AMOEBA optimization:', datetime.now() - startTime)
             
        best_theta = result_optim['x']
        chisquare = result_optim['fun']
        niter = result_optim['nfev']

        success  = result_optim['success']
        print('Note:\n-Minimization derived successfully?', success)

        message = result_optim['message']
        print('-Message:', message)

        amoebaresultdir = SAVINGDIR +  '/results_AMOEBA'
        os.makedirs(amoebaresultdir,exist_ok=True)

        if TYPE_OBS == 'polar':
            best_model = call_gen_disk_5params(best_theta)
            best_modelconvolved = convolve_fft(best_model, PSF, boundary='wrap')
            best_residuals = (REDUCED_DATA - best_modelconvolved)
            best_residuals_snr = best_residuals / NOISE

        else:
            # Generate disk model
            best_model = call_gen_disk_5params(best_theta)

            # Rotate the disk model for different angles
            # and convolve it by the PSF
            best_modelconvolved = vip.fm.cube_inject_fakedisk(best_model, PA_RANGE, psf=PSF)

            # Remove the disk from the observation
            CUBE_DIFF = REDUCED_DATA_MASK - best_modelconvolved

            # Reduce the observation
            best_residuals = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_RANGE, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=False)

            # Compute the residuals and the chisquare
            best_residuals_snr = best_residuals / NOISE

            fits.writeto(os.path.join(amoebaresultdir,'best_residuals_cube.fits'),
                     CUBE_DIFF, overwrite=True)
            

        print('\nSave files at:\n', amoebaresultdir)
        fits.writeto(os.path.join(amoebaresultdir,'best_model_amoeba.fits'),
                     best_model, overwrite=True)
        fits.writeto(os.path.join(amoebaresultdir,'best_modelconvolved_amoeba.fits'),
            best_modelconvolved, overwrite=True)
        
        #fits.writeto(os.path.join(amoebaresultdir,'best_model_fm_amoeba.fits'),
        #             best_model_fm, overwrite=True)

        fits.writeto(os.path.join(amoebaresultdir,'best_residuals_amoeba.fits'),
                     best_residuals, overwrite=True)
        
        fits.writeto(os.path.join(amoebaresultdir,'best_residuals_snr_amoeba.fits'),
            best_residuals_snr, overwrite=True)

        
        print('=== Summary ===')
        print('- If there is no disk, \nthe chisquare is %.3e' % chisquare_init_nodisk, 'i.e %.0f' % chisquare_init_nodisk)
        
        print('- The initial parameters were:\n', theta_init, '\nfor a chisquare = %.3e' % chisquare_init, 'i.e %.0f' % chisquare_init)
        
        print('- The best parameters derived are:\n', best_theta, '\nfor a chisquare of: %.2e' % chisquare, 'after', niter, 'iterations.')

        diff1 = chisquare_init_nodisk-chisquare_init
        diff2 = chisquare_init-chisquare
        diff3 = chisquare_init_nodisk-chisquare
        print("chisquare (no-disk) - chisquare (first-guess): {:.3e}, i.e., {:.0f}".format(diff1,diff1))
        print("chisquare (first-guess) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff2, diff2))
        print("chisquare (no-disk) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff3,diff3))