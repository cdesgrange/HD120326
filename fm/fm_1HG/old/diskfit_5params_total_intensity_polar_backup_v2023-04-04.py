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
from vip_hci.preproc.derotation import cube_derotate

import astro_unit_conversion as convert

from derive_noise_map import *

import pyregion
import logging

os.environ["OMP_NUM_THREADS"] = "1"

#from functions_diskfit_mcmc import *


######################################################
###################### logfile #######################
######################################################
# info
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#error
def log_error(fn, message):
    with open("{}_error_{}.log".format(fn, date), "a") as myfile:
        myfile.write(f'ERROR : {message} \n')


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
                               itilt=inc,
                               omega=argperi,
                               pxInArcsec=PIXSCALE_INS,
                               pa= -pa,
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
                                   'polar': DISK_MODEL_POLAR
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

    if TYPE_OBS == 'total_intensity':
        # Rotate the disk model for different angles and convolve it by the PSF
        modelconvolved = vip.fm.cube_inject_fakedisk(model, PA_ARRAY, psf=PSF)

        # Compute the residuals and the chisquare
        # Remove the disk from the observation
        CUBE_DIFF = (SCIENCE_DATA - modelconvolved)

        # Reduce the observation
        #im_pca, pcs, reconstr_cube, res_cube, res_cube_derot = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=False)
        im_pca = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

        res = im_pca * MASK2MINIMIZE / NOISE

    elif TYPE_OBS == 'polar':
         modelconvolved = convolve_fft(model, PSF, boundary='wrap')
         res = (REDUCED_DATA_MASK - modelconvolved) / NOISE
        
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
def from_param_to_theta_init_5params(parameters_yaml):
    """ create a initial set of parameters from the initial parameters
        store in the init yaml file
    Args:
        parameters_yaml: dic, all the parameters of the exploration algo and PCA
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    rad_init = parameters_yaml['rad_init']
    pa_init = parameters_yaml['pa_init']
    inc_init = parameters_yaml['inc_init']
    g1_init = parameters_yaml['g1_init']
    scaling = parameters_yaml['scaling']

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
def from_param_to_theta_init(parameters_yaml):
    """ create a initial set of MCMC parameter from the initial parameters
        store in the init yaml file
    Args:
        parameters_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    rad_init = parameters_yaml['rad_init']
    ain_init = parameters_yaml['ain_init']
    aout_init = parameters_yaml['aout_init']
    inc_init = parameters_yaml['inc_init']
    pa_init = parameters_yaml['pa_init']
    argperi_init = parameters_yaml['argperi_init']
    ecc_init = parameters_yaml['ecc_init']
    ksi0_init = parameters_yaml['ksi0_init']
    g1_init = parameters_yaml['g1_init']
    g2_init = parameters_yaml['g2_init']
    alpha_init = parameters_yaml['alpha_init']
    N_init = parameters_yaml['N_init']

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



## LOAD STUFF ##

def load_PSF():
    print('\n(Load PSF data)')
    if TYPE_OBS == 'polar': 
        FN_PSF_LEFT = POSTPROCESS + params_yaml['FN_PSF_LEFT']
        FN_PSF_RIGHT = POSTPROCESS + params_yaml['FN_PSF_RIGHT']
        if display:
            print('The path for the PSF (left) data:\n', DATADIR+FN_PSF_LEFT)
            print('The path for the PSF (right) data:\n', DATADIR+FN_PSF_RIGHT)
         
        PSF_LEFT = fits.getdata(os.path.join(DATADIR, FN_PSF_LEFT))
        PSF_RIGHT = fits.getdata(os.path.join(DATADIR, FN_PSF_RIGHT))
        psf = np.concatenate((PSF_LEFT, PSF_RIGHT))
        psf = np.nansum(psf, axis=0)
        psf = frame_shift(psf,0.5,0.5)

    elif TYPE_OBS == 'total_intensity':
        FN_PSF = PREPROCESS + params_yaml['FN_PSF']
        if display:
            print('The path for the PSF data:\n', DATADIR+FN_PSF)
        psf = fits.getdata(os.path.join(DATADIR, FN_PSF))
        psf = np.nansum(psf, axis=1)[IBAND]

    if CROP_PSF !=0:
        if display: print('The size of the PSF data *before* cropping is:', np.shape(psf))
        psf = psf[CROP_PSF:-CROP_PSF+1, CROP_PSF:-CROP_PSF+1]
        if display: print('The size of the PSF data *after* cropping is:', np.shape(psf))
    else:
        if display: print('The size of the PSF data is:', np.shape(psf))
        
    total_flux_psf = np.nansum(psf)
    if display: print('The total flux of the PSF image is:',total_flux_psf)
    psf = psf/total_flux_psf

     
    fits.writeto(os.path.join(inputs_resultdir,'PSF.fits'),
                     psf, overwrite=True)
    
    return psf


        
def load_SCIENCE():
    print('\n(Load SCIENCE data)')
    if TYPE_OBS == 'polar':
        FN_SCIENCE =  POSTPROCESS + params_yaml['FN_SCIENCE']
        science_data = fits.getdata(os.path.join(DATADIR, FN_SCIENCE))

    elif TYPE_OBS == 'total_intensity':
        FN_SCIENCE = PREPROCESS + params_yaml['FN_SCIENCE']
        science_data = fits.getdata(os.path.join(DATADIR, FN_SCIENCE))[IBAND]

    if display:
        print('The path for the SCIENCE data:\n', FN_SCIENCE)

    if TYPE_OBS == 'polar' and CROP_SCIENCE != 0:
        if display: print('The size of the SCIENCE data *before* cropping is:', np.shape(science_data))
        science_data = science_data[CROP_SCIENCE:-CROP_SCIENCE, CROP_SCIENCE:-CROP_SCIENCE]
        if display: print('The size of the SCIENCE data *after* cropping is:', np.shape(science_data))

    elif TYPE_OBS == 'total_intensity' and CROP_SCIENCE != 0:
        if display: print('The size of the SCIENCE data *before* cropping is:', np.shape(science_data))
        science_data = science_data[:,CROP_SCIENCE:-CROP_SCIENCE, CROP_SCIENCE:-CROP_SCIENCE]
        if display: print('The size of the SCIENCE data *after* cropping is:', np.shape(science_data))
        
    elif  CROP_SCIENCE == 0 and display: print('The size of the SCIENCE data is:', np.shape(science_data))

    if TYPE_OBS == 'polar':
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image.fits'), science_data, overwrite=True)
    #elif TYPE_OBS == 'total_intensity':
    #    fits.writeto(os.path.join(inputs_resultdir,'cube_science.fits'), science_data, overwrite=True)
        
    return science_data


def load_NOISE():
    print('\n(Load NOISE data)')
    if TYPE_OBS == 'polar':
        FN_NOISE = POSTPROCESS + params_yaml['FN_NOISE']
        if display: print('The path for the NOISE data:\n', DATADIR+FN_NOISE)
        noise_map = fits.getdata(os.path.join(DATADIR, FN_NOISE))

    elif TYPE_OBS == 'total_intensity':
        METH_NOISE = params_yaml['METH_NOISE']
        if METH_NOISE != 'compute_it':
            FN_NOISE   = PREPROCESS + params_yaml['FN_NOISE']
            if display: print('The path for the NOISE data:\n', DATADIR+FN_NOISE)
            noise_map = fits.getdata(os.path.join(DATADIR, FN_NOISE))

        if METH_NOISE == 'compute_it':
            print('The noise map is derived by applying PCA on the opposite parallactic angles.')
            NOISE_ALMOST =  vip.psfsub.pca_fullfr.pca(SCIENCE_DATA, -PA_ARRAY,
                                           ncomp=NMODES, mask_center_px=MASK_RAD,
                                           imlib='opencv', full_output=False)
            
            fits.writeto(os.path.join(inputs_resultdir,'reduced_image_opp_pa.fits'),
                     NOISE_ALMOST, overwrite=True)

            noise_map = compute_limdet_map_ann(NOISE_ALMOST, dr=2, alpha=1, center='center', even_or_odd='even')

    else: raise ValueError('(!) The parameter TYPE_OBS is misdefined, TYPE_OBS =', TYPE_OBS, '...')
    noise_map[noise_map==0] = np.nan #1e3

    if CROP_NOISE != 0:
        if display: print('The size of the NOISE data *before* cropping is:', np.shape(noise_map))
        noise_map = noise_map[CROP_NOISE:-CROP_NOISE, CROP_NOISE:-CROP_NOISE]
        if display: print('The size of the NOISE data *after* cropping is:', np.shape(noise_map))
    elif CROP_NOISE == 0 and display: print('The size of the NOISE data is:', np.shape(noise_map))

    fits.writeto(os.path.join(inputs_resultdir,'NOISE.fits'),
                     noise_map, overwrite=True)
    return noise_map



def load_PA_ARRAY():
    print('\n(Load PA data)')
    if TYPE_OBS == 'total_intensity':
        FN_PA  = PREPROCESS + params_yaml['FN_PA']
        if display: print('The path for the parallactic angles array:\n', DATADIR+FN_PA)
        pa_array = -fits.getdata(os.path.join(DATADIR,FN_PA))
        if display: print('The size of the parallactic angles array is:', np.shape(pa_array))
        return pa_array
    else:
        return None

def load_MASK2MINIMIZE():
    print('\n(Load MASK data)')
    FN_MASK = params_yaml['FN_MASK']
    if display: print('The path for the mask:\n', DATADIR+FN_MASK)
    if TYPE_OBS  == 'polar':
        mask2minimize = fits.getdata(FN_MASK)
        mask2minimize = mask2minimize[CROP_SCIENCE:-CROP_SCIENCE,CROP_SCIENCE:-CROP_SCIENCE]
    else:
        dum, header = fits.getdata(os.path.join(inputs_resultdir,'NOISE.fits'),header=True) #!!!
        file_2_ellipses = FN_MASK
        with open (file_2_ellipses, "r") as myfile:
            reg_2_ellipses=myfile.read() 
        region_2_ellipses = pyregion.parse(reg_2_ellipses)
        myfilter = region_2_ellipses.get_filter(header=header)
        mask2minimize = (myfilter[1] & ~myfilter[0]).mask((dum.shape[0],dum.shape[1]))
        mask2minimize = np.where(mask2minimize == False, 0, 1)
    
    if display: print('The size of the mask is:', np.shape(mask2minimize))
    
    fits.writeto(os.path.join(inputs_resultdir,'mask2minimize.fits'),
                     mask2minimize, overwrite=True)
    return mask2minimize


## PROCESS DATA ##
def process_SCIENCE_PCA():
    if TYPE_OBS == 'total_intensity':
        print('We reduced the data by applying PCA.')
        red_data =  vip.psfsub.pca_fullfr.pca(SCIENCE_DATA, PA_ARRAY,
                            ncomp=NMODES, mask_center_px=MASK_RAD,
                            imlib='opencv', full_output=False)

        fits.writeto(os.path.join(inputs_resultdir,'reduced_image.fits'),
                     red_data, overwrite=True)
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_mask.fits'),
                     red_data*MASK2MINIMIZE, overwrite=True)
        return red_data
    return None
    


def apply_MASK2SCIENCE():
    if TYPE_OBS == 'polar':
        SCIENCE_DATA_MASK = np.copy(SCIENCE_DATA) * MASK2MINIMIZE
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_mask.fits'),
                     SCIENCE_DATA_MASK, overwrite=True)
        
    elif TYPE_OBS == 'total_intensity':
        MASK2MINIMIZE_CUBE = np.repeat(MASK2MINIMIZE[np.newaxis, :, :], len(PA_ARRAY), axis=0)
        MASK2MINIMIZE_CUBE_PA = cube_derotate(MASK2MINIMIZE_CUBE, PA_ARRAY, imlib='opencv', interpolation='nearneig')
        SCIENCE_DATA_MASK = np.copy(SCIENCE_DATA) * MASK2MINIMIZE_CUBE_PA
        #fits.writeto(os.path.join(inputs_resultdir,'cube_mask_science.fits'), SCIENCE_DATA_MASK, overwrite=True)
        #fits.writeto(os.path.join(inputs_resultdir,'cube_mask.fits'), MASK2MINIMIZE_CUBE_PA, overwrite=True)
    return SCIENCE_DATA_MASK


## FINAL TESTS ##

def do_test_disk_empty():
    print('\n- No disk')
    startTime = datetime.now()
    theta_init_no_disk = np.copy(THETA_INIT)
    theta_init_no_disk[-1] = 0
    if display: print('Check disk model parameters without flux (scaling_flux should be set to 0):\n', theta_init_no_disk)
        
    if exploration_algo == "MCMC":
        lnpb_model = lnpb(theta_init_no_disk)
        print("Test likelihood on initial model:", lnpb_model)
             
    elif exploration_algo == "AMOEBA":
        chisquare_init_nodisk = chisquare_5params(theta_init_no_disk)
        #print("Test chisquare on initial model: {:.0f}".format(chisquare_init_nodisk))

    print("Time for a single model: ", datetime.now() - startTime)
    return chisquare_init_nodisk


def do_test_disk_first_guess():
    print('\n- Initial model')
    startTime = datetime.now()
    print('Parameter starting point:', THETA_INIT)
        
    if exploration_algo == "MCMC":
        lnpb_model = lnpb(THETA_INIT)
        print("Test likelihood on initial model:", lnpb_model)
        suffix = '_{:.0f}'.format(lnpb_model)
             
    elif exploration_algo == "AMOEBA" and 0:
        chisquare_init = chisquare_5params(THETA_INIT)
        #print("Test chisquare on initial model: {:.0f}".format(chisquare_init))
        suffix = '_{:.0f}.'.format(chisquare_init)
        print("chisquare (no-disk) - chisquare (first-guess): {:.0f}".format((chisquare_init_nodisk-chisquare_init)))

    if TYPE_OBS == 'polar':
        model = call_gen_disk_5params(THETA_INIT)
        modelconvolved = convolve_fft(model, PSF, boundary='wrap')
        best_residuals = (SCIENCE_DATA - modelconvolved)
        best_residuals_snr = best_residuals / NOISE

    else:
        # Generate disk model
        model = call_gen_disk_5params(THETA_INIT)

        # Rotate the disk model for different angles
        # and convolve it by the PSF
        modelconvolved = vip.fm.cube_inject_fakedisk(model, PA_ARRAY, psf=PSF)
        modelconvolved0 = convolve_fft(model, PSF, boundary='wrap')
            
        # Remove the disk from the observation
        CUBE_DIFF = SCIENCE_DATA - modelconvolved

        # Reduce the observation
        #best_residuals = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)
        best_residuals, pcs, reconstr_cube, res_cube, res_cube_derot = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_PCA_pcs_cube.fits'),
                     pcs, overwrite=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_PCA_reconstr_cube.fits'),
                     reconstr_cube, overwrite=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_PCA_res_cube.fits'),
                     res_cube, overwrite=True)
        fits.writeto(os.path.join(firstguess_resultdir,'first_guess_PCA_res_cube_derot.fits'),
                     res_cube_derot, overwrite=True)
        
        # Compute the residuals and the chisquare
        best_residuals_snr = best_residuals / NOISE
        best_residuals_snr_masked = best_residuals_snr * MASK2MINIMIZE
        
        chisquare_init = np.nansum(best_residuals_snr_masked * best_residuals_snr_masked)
        suffix = '_{:.0f}'.format(chisquare_init)
        print("chisquare (no-disk) - chisquare (first-guess): {:.0f}".format((chisquare_init_nodisk-chisquare_init)))

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
    return chisquare_init
    

    
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

    
    fn_log = "{}/results/logs/log_diskfit_{}".format(os.getcwd(),  str_yalm[:-6] )
    fn_log_info = "{}_info_{}.log".format(fn_log, date)
    sys.stdout = Logger(fn_log_info)
    print("Write a logfile with all printed infos at", fn_log_info)

    # Open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files', str_yalm)
    if display: print('\nThe configuration file .yaml is located at:\n', yaml_path_file)
    with open(yaml_path_file, 'r') as yaml_file:
        params_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
        TYPE_OBS = params_yaml['TYPE_OBS']

    ## Initialize paths
    ## Paths
    DATADIR = params_yaml['DATADIR']
    PREPROCESS = params_yaml['PREPROCESS']
    POSTPROCESS = params_yaml['POSTPROCESS']

    # Saving directories
    SAVINGDIR = params_yaml['SAVINGDIR'] + date
    os.makedirs(SAVINGDIR,exist_ok=True)
    inputs_resultdir = SAVINGDIR + '/inputs'
    os.makedirs(inputs_resultdir,exist_ok=True)
    firstguess_resultdir = SAVINGDIR + '/first_guess'
    os.makedirs(firstguess_resultdir,exist_ok=True)
    
    print('\nSave input files at:\n', inputs_resultdir)
    print('\nSave first guess files at:\n', firstguess_resultdir)
    
    ## Initialize variables
    # System
    DISTANCE_STAR = params_yaml['DISTANCE_STAR']

    # Observation
    PIXSCALE_INS = params_yaml['PIXSCALE_INS']
    IBAND = params_yaml['IBAND']
    OWA = params_yaml['OWA'] # OWA = dimension before cropping

    # Processing
    CROP_SCIENCE =  params_yaml['CROP_SCIENCE']
    CROP_NOISE =  params_yaml['CROP_NOISE']
    CROP_PSF =  params_yaml['CROP_PSF']
    NMODES = params_yaml['NMODES']
    DIMENSION = int(OWA - CROP_SCIENCE*2) # DIMENSION  = dimension after cropping
    MASK_RAD =  params_yaml['MASK_RAD']
    NORM =  params_yaml['SCIENCE_NORM']

    # Modelling
    exploration_algo = params_yaml['exploration_algo']
    THETA_INIT = from_param_to_theta_init_5params(params_yaml)
    NOISE_MULTIPLICATION_FACTOR = params_yaml['NOISE_MULTIPLICATION_FACTOR']
    PARAMS_TO_FIT = params_yaml['PARAMS_TO_FIT']
    if TYPE_OBS == 'polar':
        DISK_MODEL_POLAR = True
    else:
        DISK_MODEL_POLAR = False
    
    if display:
        print('\n(!) Check the parameters (!)')
        print('\n- System params:')
        print('The star is located at',  DISTANCE_STAR, 'pc.')
        
        print('\n- Observational params:')
        print('The observation is in', TYPE_OBS, '.')
        if TYPE_OBS == 'total_intensity':
            print(NMODES, 'modes are used to apply the PCA.')
            print('We consider the channel', IBAND+1, '.')
        print('The pixel scale is', PIXSCALE_INS, 'as/pix.')
        print('Initially, the field of view is:', OWA*PIXSCALE_INS, 'arsec, i.e., ', OWA, 'pixels.')
        print('After cropping, the field of view is:', DIMENSION*PIXSCALE_INS, 'arsec, i.e., ', DIMENSION, 'pixels.')
        print('The cropping parameters are twice:',CROP_SCIENCE, '(science),', CROP_PSF, '(psf), and', CROP_NOISE, '(noise).')  
        
        print('\n- Processing params:')
        print('The mask radius used is:', MASK_RAD)      
        print('The normalization factor used is:', NORM)
        print('The exploration algo used is:', exploration_algo)      


    if display:
        print('\n- Load the PSF, (PA_ARRAY), SCIENCE, NOISE, and MASK data ')
    
    ## Load data
    # PSF
    PSF = load_PSF()
    
    # PA (if needed, i.e., only for pupil-mode observations)
    PA_ARRAY = load_PA_ARRAY()
    
    # Science
    SCIENCE_DATA = load_SCIENCE()

    # Noise
    NOISE = load_NOISE()
    
    # Define/Load the mask within the residuals will be minimized in the science - model data
    MASK2MINIMIZE = load_MASK2MINIMIZE()
    
    ## Processing
    # Process SCIENCE data (only if we do PCA forward modelling)
    RED_DATA = process_SCIENCE_PCA()

    RED_DATA_PA = np.repeat(RED_DATA[np.newaxis, :, :], len(PA_ARRAY), axis=0)
    RED_DATA_PA = cube_derotate(RED_DATA_PA, PA_ARRAY, imlib='opencv',  #interpolation='nearneig'
                                )

    fits.writeto(os.path.join(inputs_resultdir,'reduced_image_cube.fits'),
                     RED_DATA_PA, overwrite=True)

    SCIENCE_DATA = np.copy(RED_DATA_PA)
        
    # We multiply the SCIENCE data by the mask2minimize
    SCIENCE_DATA_MASK = apply_MASK2SCIENCE()


    ## TESTS ## 
    print('\n\n=== Make final test before running the MCMC / AMOEBA exploration algorithm ===')
    # What is the chisquare if the disk model is empty?
    chisquare_init_nodisk = do_test_disk_empty()
        
    # What is the chisquare / likelihood of the initial model?
    chisquare_init = do_test_disk_first_guess()

        
    if exploration_algo == "MCMC":
        print('\n\n=== Initialization MCMC ===')
        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(params_yaml)

        # load the Parameters necessary to launch the MCMC
        NWALKERS = params_yaml['NWALKERS']  #Number of walkers
        N_ITER_MCMC = params_yaml['N_ITER_MCMC']  #Number of interation
        N_DIM_MCMC = params_yaml['N_DIM_MCMC']  #Number of MCMC dimension

        # last chance to remove some global variable to be as light as possible
        # in the MCMC
        del params_yaml

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
        bounds = [(10,100), (0,180), (0, 89), (0,1), (0,None)] #params_yaml['bounds']
        
        print('The parameters to fit are:', PARAMS_TO_FIT)
        print('(!) The initial guess parameters are:', THETA_INIT)
        print('The bounds are:', bounds)

        print(bounds, type(bounds))
        a = zip(*bounds)
        print(a)
        result_optim = op.minimize(chisquare_5params,
                                   THETA_INIT,
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
            best_residuals = (SCIENCE_DATA - best_modelconvolved)
            best_residuals_snr = best_residuals / NOISE

        else:
            # Generate disk model
            best_model = call_gen_disk_5params(best_theta)

            # Rotate the disk model for different angles
            # and convolve it by the PSF
            best_modelconvolved = vip.fm.cube_inject_fakedisk(best_model, PA_ARRAY, psf=PSF)

            # Remove the disk from the observation
            CUBE_DIFF = SCIENCE_DATA - best_modelconvolved

            # Reduce the observation
            best_residuals = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=False)

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
        
        print('- The initial parameters were:\n', THETA_INIT, '\nfor a chisquare = %.3e' % chisquare_init, 'i.e %.0f' % chisquare_init)
        
        print('- The best parameters derived are:\n', best_theta, '\nfor a chisquare of: %.2e' % chisquare, 'after', niter, 'iterations.')

        diff1 = chisquare_init_nodisk-chisquare_init
        diff2 = chisquare_init-chisquare
        diff3 = chisquare_init_nodisk-chisquare
        print("chisquare (no-disk) - chisquare (first-guess): {:.3e}, i.e., {:.0f}".format(diff1,diff1))
        print("chisquare (first-guess) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff2, diff2))
        print("chisquare (no-disk) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff3,diff3))