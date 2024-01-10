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
import shutil
import argparse

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
    polar = theta[5]

    print(polar)
     
    # Fixed parameters
    ain = AIN #12#10
    aout = AOUT #-1.9 #-5 
   
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
                                   'polar': polar,
                               })
    
    return model.compute_scattered_light() * scaling



########################################################
##################### CHISQUARE ########################
########################################################

def chisquare_params(theta):
    if NOBS==1:
        Chisquare = chisquare_params_1obs(theta)
    elif NOBS==2 and np.any( np.array(TYPE_OBS_ALL) == 'polar' ):
        Chisquare = chisquare_params_2obs_polar(theta)
    elif NOBS==3 and np.all( np.array(TYPE_OBS_ALL) == 'total_intensity' ):
        Chisquare = chisquare_params_3obs_ti_only(theta)
    elif NOBS==4 and np.any( np.array(TYPE_OBS_ALL) == 'polar' ):
        Chisquare = chisquare_params_4obs_polar(theta)
    else:
        raise ValueError('Check the input "NOBS" ({}) and "TYPE_OBS_ALL" ({}) values'.format(NOBS, TYPE_OBS_ALL))
    return Chisquare



########################################################

def chisquare_params_4obs_polar(theta):
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
    Chisquare_red_final = []
    theta0 = [theta[0],theta[1],theta[2],theta[3],theta[4],False]
    theta1 = [theta[0],theta[1],theta[2],theta[3],theta[5],False]
    theta2 = [theta[0],theta[1],theta[2],theta[3],theta[6],False]
    theta3 = [theta[0],theta[1],theta[2],theta[3],theta[7],True]
    
    ## Total intensity BBH ##
    i=0
    model = call_gen_disk_5params(theta0)
            
    # Rotate the disk model for different angles and convolve it by the PSF
    modelconvolved = vip.fm.cube_inject_fakedisk(model, -PA_ARRAY[i], psf=PSF[i])

    # Compute the residuals and the chisquare
    # Remove the disk from the observation
    CUBE_DIFF = (SCIENCE_DATA[i] - modelconvolved)

    # Reduce the observation
    #im_pca, pcs, reconstr_cube, res_cube, res_cube_derot = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY, ncomp=NMODES, mask_center_px=MASK_RAD, full_output=False)
    im_pca = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY[i], ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

    res_snr_all = im_pca / NOISE[i]
    res_snr_mask = res_snr_all * MASK2MINIMIZE 
    
    Chisquare = np.nansum(res_snr_mask**2)
    Chisquare_red = Chisquare/(NRESEL-NPARAMS-1)
    Chisquare_red_final.append(Chisquare_red)
            
    if SAVE_INTERMEDIATE_RESULTS:
        dt =  datetime.now()  - startTime
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')

        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_1{}.fits'.format(suffix)), im_pca, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_snr_1{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'disk_model_convolved_1{}.fits'.format(suffix)), modelconvolved0, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'theta_1{}.fits'.format(suffix)), theta, overwrite=True)

    if SAVE_DETAIL_RESULTS:
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')
        DIFF0 = RED_DATA - modelconvolved0
            
        fits.writeto(os.path.join(detail_resultdir,'disk_model_1{}.fits'.format(suffix)),model, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'disk_model_convolved_1{}.fits'.format(suffix)),modelconvolved0, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_1{}.fits'.format(suffix)), DIFF0, overwrite=True)    

        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_1{}.fits'.format(suffix)), im_pca, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_snr_1{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'theta_1{}.fits'.format(suffix)), theta, overwrite=True)
            
    print('For theta =', theta, '\n-> Chisquare = {:.4e} i.e. {:.0f}, Reduced chisquare =  {:.2f} (total intensity 1)'.format(Chisquare, Chisquare, Chisquare_red))


    ##  Total intensity H23 1 ##
    i=1
    model = call_gen_disk_5params(theta1)
            
    # Rotate the disk model for different angles and convolve it by the PSF
    modelconvolved = vip.fm.cube_inject_fakedisk(model, -PA_ARRAY[i], psf=PSF[i])

    # Remove the disk from the observation
    CUBE_DIFF = (SCIENCE_DATA[i] - modelconvolved)

    # Reduce the observation
    im_pca = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY[i], ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

    res_snr_all = im_pca / NOISE[i]
    res_snr_mask = res_snr_all * MASK2MINIMIZE 

    # Compute the residuals and the chisquare
    Chisquare = np.nansum(res_snr_mask**2)
    Chisquare_red = Chisquare/(NRESEL-NPARAMS-1)
    Chisquare_red_final.append(Chisquare_red)
            
    if SAVE_INTERMEDIATE_RESULTS:
        dt =  datetime.now()  - startTime
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')

        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_2{}.fits'.format(suffix)), im_pca, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_snr_2{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'disk_model_convolved_2{}.fits'.format(suffix)), modelconvolved0, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'theta_2{}.fits'.format(suffix)), theta, overwrite=True)

    if SAVE_DETAIL_RESULTS:
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')
        DIFF0 = RED_DATA - modelconvolved0
            
        fits.writeto(os.path.join(detail_resultdir,'disk_model_2{}.fits'.format(suffix)),model, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'disk_model_convolved_2{}.fits'.format(suffix)),modelconvolved0, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_2{}.fits'.format(suffix)), DIFF0, overwrite=True)    

        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_2{}.fits'.format(suffix)), im_pca, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_snr_2{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'theta_2{}.fits'.format(suffix)), theta, overwrite=True)
            
    print('For theta =', theta, '\n-> Chisquare = {:.4e} i.e. {:.0f}, Reduced chisquare =  {:.2f} (total intensity 2)'.format(Chisquare, Chisquare, Chisquare_red))

    ##  Total intensity H23 2 ##
    i=2
    model = call_gen_disk_5params(theta2)
            
    # Rotate the disk model for different angles and convolve it by the PSF
    modelconvolved = vip.fm.cube_inject_fakedisk(model, -PA_ARRAY[i], psf=PSF[i])

    # Remove the disk from the observation
    CUBE_DIFF = (SCIENCE_DATA[i] - modelconvolved)

    # Reduce the observation
    im_pca = vip.psfsub.pca_fullfr.pca(CUBE_DIFF, PA_ARRAY[i], ncomp=NMODES, mask_center_px=MASK_RAD, imlib='opencv', full_output=False)

    res_snr_all = im_pca / NOISE[i]
    res_snr_mask = res_snr_all * MASK2MINIMIZE 

    # Compute the residuals and the chisquare
    Chisquare = np.nansum(res_snr_mask**2)
    Chisquare_red = Chisquare/(NRESEL-NPARAMS-1)
    Chisquare_red_final.append(Chisquare_red)
            
    if SAVE_INTERMEDIATE_RESULTS:
        dt =  datetime.now()  - startTime
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')

        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_3{}.fits'.format(suffix)), im_pca, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_snr_3{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'disk_model_convolved_3{}.fits'.format(suffix)), modelconvolved0, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'theta_3{}.fits'.format(suffix)), theta, overwrite=True)

    if SAVE_DETAIL_RESULTS:
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        modelconvolved0 = convolve_fft(model, PSF[i], boundary='wrap')
        DIFF0 = RED_DATA - modelconvolved0
            
        fits.writeto(os.path.join(detail_resultdir,'disk_model_3{}.fits'.format(suffix)),model, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'disk_model_convolved_3{}.fits'.format(suffix)),modelconvolved0, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_3{}.fits'.format(suffix)), DIFF0, overwrite=True)    

        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_3{}.fits'.format(suffix)), im_pca, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_snr_3{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'theta_3{}.fits'.format(suffix)), theta, overwrite=True)

    print('For theta =', theta, '\n-> Chisquare = {:.4e} i.e. {:.0f}, Reduced chisquare =  {:.2f} (total intensity 3)'.format(Chisquare, Chisquare, Chisquare_red))
    

    ## Polar BBH ##
    i=3
    model = call_gen_disk_5params(theta3)
    modelconvolved = convolve_fft(model, PSF[i], boundary='wrap')
    res_all = (SCIENCE_DATA[i] - modelconvolved) #(SCIENCE_DATA_MASK - modelconvolved0)
    res_snr_all  = res_all / NOISE[i]
    res_snr_mask = res_snr_all * MASK2MINIMIZE 
    Chisquare = np.nansum(res_snr_mask**2)
    Chisquare_red = Chisquare/(NRESEL-NPARAMS-1)
    Chisquare_red_final.append(Chisquare_red)

    if SAVE_INTERMEDIATE_RESULTS:
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_4{}.fits'.format(suffix)), res_all, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'residuals_fm_snr_4{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'disk_model_convolved_4{}.fits'.format(suffix)), modelconvolved, overwrite=True)
        fits.writeto(os.path.join(intermediate_resultdir,'theta_4{}.fits'.format(suffix)), theta, overwrite=True)

    if SAVE_DETAIL_RESULTS:
        suffix = "_{:.2f}min_{:.0f}_{:.2f}".format(dt.seconds/60, Chisquare, Chisquare_red)
        fits.writeto(os.path.join(detail_resultdir,'disk_model_4{}.fits'.format(suffix)),model, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'disk_model_convolved_4{}.fits'.format(suffix)),modelconvolved, overwrite=True)      
        fits.writeto(os.path.join(detail_resultdir,'residuals_4{}.fits'.format(suffix)), res_all, overwrite=True)    
        fits.writeto(os.path.join(detail_resultdir,'residuals_fm_snr_4{}.fits'.format(suffix)), res_snr_all, overwrite=True)
        fits.writeto(os.path.join(detail_resultdir,'theta_4{}.fits'.format(suffix)), theta, overwrite=True)


    Chisquare_red_final = ( (Chisquare_red_final[0] + Chisquare_red_final[1] + Chisquare_red_final[2]) / 3 + Chisquare_red_final[3] ) /2  
    print('For theta =', theta, '\n-> Chisquare = {:.4e} i.e. {:.0f}, Reduced chisquare =  {:.2f} (polar)'.format(Chisquare, Chisquare, Chisquare_red))
    print('(!) Final reduced chisquare =  {:.2f}\n'.format(Chisquare_red_final))

    return Chisquare_red_final



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
def from_param_to_theta_init_8params(parameters_yaml):
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
    scaling1 = parameters_yaml['scaling1']
    scaling2 = parameters_yaml['scaling2']
    scaling3 = parameters_yaml['scaling3']
    scaling4 = parameters_yaml['scaling4']

    #theta_init = (rad_init, pa_init, np.cos(np.radians(inc_init)), g1_init, scaling)
    theta_init = (rad_init, pa_init, inc_init, g1_init, scaling1, scaling2, scaling3, scaling4)
    

    for parami in theta_init:
        if parami == 0:
            raise ValueError("""Do not initialize one of your parameters
            at exactly 0.0, it messes up the small ball at MCMC initialization"""
                             )
    return np.array(theta_init)




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


################################
########## LOAD STUFF ##########
################################
def load_PSF_final():
    '''
    Return the normalized PSF of the dataset, or the different datasets considered.
    Use the function load_one_PSF().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        POSTPROCESS = POSTPROCESS_ALL
        PREPROCESS = PREPROCESS_ALL
        FN_PSF =  FN_PSF_ALL
        CROP_PSF = CROP_PSF_ALL
        psf = load_one_PSF(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_PSF, ID='only')
        
    else:
        psf = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            POSTPROCESS = POSTPROCESS_ALL[i]
            PREPROCESS = PREPROCESS_ALL[i]
            FN_PSF =  FN_PSF_ALL[i]
            CROP_PSF = CROP_PSF_ALL[i]
            psf.append( load_one_PSF(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_PSF, CROP_PSF, ID=str(i)) )

    return np.array(psf)
        
        
def load_one_PSF(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_PSF, CROP_PSF, ID=None):
    '''
    Return the normalized PSF of the dataset. The PSF is defined as the sum of the PSF.
    The PSF is then normalized by the sum of the pixel values.
    This function differs if the data are acquired in polarised light or total intensity light.
    Save the PSF file in the path 'inputs_resultdir'.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .DATA_DIR (string): common prefix path
    .PRE_PROCESS (string): folder in which is located the PSF (for total intensity data)
    .POST_PROCESS (string): folder in which is located the PSF (for polar data)
    .params_yaml (yaml file): configuration file in which some parameters should be defined (FN_PSF, FN_PSF_LEFT, and/or FN_PSF_RIGHT)
    .IBAND (int): index of the band to consider
    .CROP_PSF (int): cropping value apply to the left, right, top, and botton of the image
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files
    '''
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
        FN_PSF_EXT = PREPROCESS + FN_PSF
        if display:
            print('The path for the PSF data:\n', DATADIR+FN_PSF_EXT)
        psf = fits.getdata(os.path.join(DATADIR, FN_PSF_EXT))
        if len(np.shape(psf)) == 4: psf = np.nanmean(psf, axis=1)
        # sum channels (if COMBINED set to True), otherwise consider one channel
        if COMBINED: psf = np.nansum(psf, axis=0)
        else: psf=psf[IBAND]

    if CROP_PSF !=0:
        if display: print('The size of the PSF data *before* cropping is:', np.shape(psf))
        psf = psf[CROP_PSF:-CROP_PSF+1, CROP_PSF:-CROP_PSF+1]
        if display: print('The size of the PSF data *after* cropping is:', np.shape(psf))
    else:
        if display: print('The size of the PSF data is:', np.shape(psf))
        
    total_flux_psf = np.nansum(psf)
    if display: print('The total flux of the PSF image is:',total_flux_psf)
    psf = psf/total_flux_psf
    
    # Save file
    fits.writeto(os.path.join(inputs_resultdir,'PSF{}.fits'.format('_'+ID)),
                     psf, overwrite=True)
    
    return psf



def load_SCIENCE_final():
    '''
    Return the SCIENCE data of the dataset, or the different datasets considered.
    Use the function load_one_SCIENCE().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        POSTPROCESS = POSTPROCESS_ALL
        PREPROCESS = PREPROCESS_ALL
        FN_SCIENCE =  FN_SCIENCE_ALL
        CROP_SCIENCE = CROP_SCIENCE_ALL
        science_data = load_one_SCIENCE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_SCIENCE, CROP_SCIENCE, ID='only')
        
    else:
        science_data = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            POSTPROCESS = POSTPROCESS_ALL[i]
            PREPROCESS = PREPROCESS_ALL[i]
            FN_SCIENCE =  FN_SCIENCE_ALL[i]
            CROP_SCIENCE = CROP_SCIENCE_ALL[i]
            science_data.append( load_one_SCIENCE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_SCIENCE, CROP_SCIENCE, ID=str(i)) )

    return np.array(science_data)
        


def load_one_SCIENCE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_SCIENCE, CROP_SCIENCE, ID=None):
    '''
    Return the science data of the dataset. 
    This function differs if the data are acquired in polarised light or total intensity light.
    Save the science data in the path 'inputs_resultdir'.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .DATA_DIR (string): common prefix path
    .PRE_PROCESS (string): folder in which is located the PSF (for total intensity data)
    .POST_PROCESS (string): folder in which is located the PSF (for polar data)
    .params_yaml (yaml file): configuration file in which some parameters should be defined (FN_SCIENCE)
    .IBAND (int): index of the band to consider
    .CROP_SCIENCE (int): cropping value apply to the left, right, top, and bottom of the image
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files
    '''
    print('\n(Load SCIENCE data)')
    if TYPE_OBS == 'polar':
        FN_SCIENCE_EXT =  POSTPROCESS + FN_SCIENCE
        science_data = fits.getdata(os.path.join(DATADIR, FN_SCIENCE_EXT))

    elif TYPE_OBS == 'total_intensity':
        FN_SCIENCE_EXT = PREPROCESS + FN_SCIENCE
        if COMBINED: science_data = np.nansum(fits.getdata(os.path.join(DATADIR, FN_SCIENCE_EXT)), axis=0)
        else: science_data = fits.getdata(os.path.join(DATADIR, FN_SCIENCE_EXT))[IBAND]

    if display:
        print('The path for the SCIENCE data:\n', FN_SCIENCE_EXT)

    if TYPE_OBS == 'polar' and CROP_SCIENCE != 0:
        if display: print('The size of the SCIENCE data *before* cropping is:', np.shape(science_data))
        science_data = science_data[CROP_SCIENCE:-CROP_SCIENCE, CROP_SCIENCE:-CROP_SCIENCE]
        if display: print('The size of the SCIENCE data *after* cropping is:', np.shape(science_data))

    elif TYPE_OBS == 'total_intensity' and CROP_SCIENCE != 0:
        if display: print('The size of the SCIENCE data *before* cropping is:', np.shape(science_data))
        science_data = science_data[:,CROP_SCIENCE:-CROP_SCIENCE, CROP_SCIENCE:-CROP_SCIENCE]
        if display: print('The size of the SCIENCE data *after* cropping is:', np.shape(science_data))
        
    elif  CROP_SCIENCE == 0 and display: print('The size of the SCIENCE data is:', np.shape(science_data))

    ##if TYPE_OBS == 'polar':
    ##    fits.writeto(os.path.join(inputs_resultdir,'reduced_image.fits'), science_data, overwrite=True)
    #elif TYPE_OBS == 'total_intensity':
    #    fits.writeto(os.path.join(inputs_resultdir,'cube_science.fits'), science_data, overwrite=True)
    fits.writeto(os.path.join(inputs_resultdir,'reduced_image{}.fits'.format('_'+ID)), science_data, overwrite=True)  
    return science_data



def load_NOISE_final():
    '''
    Return the NOISE data of the dataset, or the different datasets considered.
    Use the function load_one_NOISE().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        POSTPROCESS = POSTPROCESS_ALL
        PREPROCESS = PREPROCESS_ALL
        FN_NOISE =  FN_NOISE_ALL
        CROP_NOISE = CROP_NOISE_ALL
        PA = PA_ARRAY
        SCIENCE =  SCIENCE_DATA
        noise_map = load_one_NOISE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_NOISE, CROP_NOISE, SCIENCE, PA, ID='only')
        
    else:
        noise_map = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            POSTPROCESS = POSTPROCESS_ALL[i]
            PREPROCESS = PREPROCESS_ALL[i]
            FN_NOISE =  FN_NOISE_ALL[i]
            CROP_NOISE = CROP_NOISE_ALL[i]
            PA = PA_ARRAY[i]
            SCIENCE =  SCIENCE_DATA[i]
            noise_map.append( load_one_NOISE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_NOISE, CROP_NOISE, SCIENCE, PA, ID=str(i)) )

    return np.array(noise_map)
       

def load_one_NOISE(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_NOISE, CROP_NOISE, SCIENCE, PA, ID=None):
    '''
    Return the noise map of the dataset. 
    This function differs if the data are acquired in polarised light or total intensity light.
    Save the noise map file in the path 'inputs_resultdir'.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .DATA_DIR (string): common prefix path
    .PRE_PROCESS (string): folder in which is located the PSF (for total intensity data)
    .POST_PROCESS (string): folder in which is located the PSF (for polar data)
    .params_yaml (yaml file): configuration file in which some parameters should be defined (FN_NOISE, or METH_NOISE)
    .CROP_NOISE (int): cropping value apply to the left, right, top, and bottom of the image
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files

    (optional -  only need to be defined if TYPE_OBS set to 'total_intensity' and METH_NOISE set to 'compute_it', as
    in this case the NOISE map should be derived based on the SCIENCE_DATA cube processed by PCA with opposite parallactic angles.)
    .SCIENCE_DATA (2D array if TYPE_OBS = polar, 3D array if TYPE_OBS = total_intensity): 
    .PA_ARRAY (1D array): parallactic angles
    .NMODES (int): number of modes/components to use when applying PCA 
    .MASK_RAD (int): radius of the mask corresponding approximately to the inner working angle of the coronagraph
    '''
    print('\n(Load NOISE data)')
    if TYPE_OBS == 'polar':
        FN_NOISE_EXT = POSTPROCESS + FN_NOISE
        if display: print('The path for the NOISE data:\n', DATADIR+FN_NOISE)
        noise_map = fits.getdata(os.path.join(DATADIR, FN_NOISE_EXT))

    elif TYPE_OBS == 'total_intensity':
        METH_NOISE = params_yaml['METH_NOISE']
        if METH_NOISE != 'compute_it':
            FN_NOISE_EXT = PREPROCESS + FN_NOISE
            if display: print('The path for the NOISE data:\n', DATADIR+FN_NOISE)
            noise_map = fits.getdata(os.path.join(DATADIR, FN_NOISE_EXT))

        elif METH_NOISE == 'compute_it':
            print('The noise map is derived by applying PCA on the opposite parallactic angles.')
            NOISE_ALMOST =  vip.psfsub.pca_fullfr.pca(SCIENCE, -PA,
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

    # Save file
    fits.writeto(os.path.join(inputs_resultdir,'NOISE{}.fits'.format('_'+ID)), noise_map, overwrite=True)
    return noise_map


def load_PA_ARRAY_final():
    '''
    Return the parallactic angles array for one or several pupil-tracking observations.
    Use the function load_one_PA_ARRAY().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        PREPROCESS = PREPROCESS_ALL
        FN_PA =  FN_PA_ALL
        psf = load_one_PA_ARRAY(TYPE_OBS, POSTPROCESS, PREPROCESS, FN_PSF, ID='only')
        
    else:
        psf = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            PREPROCESS = PREPROCESS_ALL[i]
            FN_PA =  FN_PA_ALL[i]
            psf.append( load_one_PA_ARRAY(TYPE_OBS, PREPROCESS, FN_PA, ID=str(i)) )

    return np.array(psf)


def load_one_PA_ARRAY(TYPE_OBS, PREPROCESS, FN_PA, ID=None):
    '''
    Return the parallactic angles array for pupil-tracking observations (assumed to be acquired
    in 'total_intensity' and not 'polar' light).
    Otherwise, return None.
    Save the parallactic angles file in the path 'inputs_resultdir'.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .DATA_DIR (string): common prefix path
    .PRE_PROCESS (string): folder in which is located the PSF (for total intensity data)
    .params_yaml (yaml file): configuration file in which some parameters should be defined (FN_PA)
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files
    '''
    print('\n(Load PA data)')
    if TYPE_OBS == 'total_intensity':
        FN_PA_EXT  = PREPROCESS + FN_PA
        if display: print('The path for the parallactic angles array:\n', DATADIR+FN_PA_EXT)
        pa_array = -fits.getdata(os.path.join(DATADIR,FN_PA_EXT))
        if display: print('The size of the parallactic angles array is:', np.shape(pa_array))
 
        # Save file
        fits.writeto(os.path.join(inputs_resultdir,'PA{}.fits'.format('_'+ID)),
                     pa_array, overwrite=True)
                 
        return pa_array
    else:
        return None

    
def load_MASK2MINIMIZE():
    '''
    Return the region where the disk model should match the SCIENCE_DATA. 
    This function differs if the data are acquired in polarised light or total intensity light.
    Save the mask map in the path 'inputs_resultdir'.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .params_yaml (yaml file): configuration file in which some parameters should be defined (FN_MASK)
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files

    if TYPE_OBS set to 'polar':
    .CROP_SCIENCE (int): cropping value apply to the left, right, top, and bottom of the image

    if TYPE_OBS set to 'total_intensity':
    .file located at the path inputs_resultdir+'NOISE.fits' should be defined. It is used to get its header,
    and more precisely the dimension of image.
    '''
    print('\n(Load MASK data)')
    FN_MASK = params_yaml['FN_MASK']
    if display: print('The path for the mask:\n', DATADIR+FN_MASK)
    
    #if TYPE_OBS  == 'polar' :
    #    mask2minimize = fits.getdata(FN_MASK)
    #    #mask2minimize = mask2minimize[CROP_SCIENCE:-CROP_SCIENCE,CROP_SCIENCE:-CROP_SCIENCE]
    #else:
    if 1:
        path = glob(inputs_resultdir+'/NOISE*.fits')[0]
        print(inputs_resultdir+'/NOISE*.fits')
        print(path)
        #dum, header = fits.getdata(os.path.join(inputs_resultdir,'NOISE*.fits'),header=True) #!!!
        dum, header = fits.getdata(path,header=True) #!!!
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
def process_SCIENCE_PCA_final():
    '''
    Return the PCA-reduced data for pupil-tracking observation(s).
    Use the function process_one_SCIENCE_PCA().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        PA = PA_ARRAY
        SCIENCE =  SCIENCE_DATA
        red_data = process_one_SCIENCE_PCA(TYPE_OBS, SCIENCE, PA, ID='only')
        
    else:
        red_data = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            PA = PA_ARRAY[i]
            SCIENCE =  SCIENCE_DATA[i]
            red_data.append( process_one_SCIENCE_PCA(TYPE_OBS, SCIENCE, PA, ID=str(i)) )

    return np.array(red_data)
        

def process_one_SCIENCE_PCA(TYPE_OBS, SCIENCE, PA, ID=None):
    '''
    Return the PCA-reduced data for a pupil-tracking observation (assumed to be acquired in 'total_intensity' and not 'polar' light).
    Otherwise, return None.
    Save at the path 'inputs_resultdir' the reduced data in different flavours: only the image, the image masked, or the cube rotating at the different parallactic angles.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .SCIENCE_DATA (2D array if TYPE_OBS = polar, 3D array if TYPE_OBS = total_intensity): 
    .PA_ARRAY (1D array): parallactic angles
    .NMODES (int): number of modes/components to use when applying PCA 
    .MASK_RAD (int): radius of the mask corresponding approximately to the inner working angle of the coronagraph
    .MASK2MINIMIZE (2D array): region where the disk model should match the SCIENCE_DATA. 
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files
    '''
    if TYPE_OBS == 'total_intensity':
        print('\n(Process the SCIENCE data)')
        print('We reduced the data by applying PCA.')
        red_data =  vip.psfsub.pca_fullfr.pca(np.copy(SCIENCE), PA,
                            ncomp=NMODES, mask_center_px=MASK_RAD,
                            imlib='opencv', full_output=False)

        fits.writeto(os.path.join(inputs_resultdir,'reduced_image{}.fits'.format('_'+ID)),
                     red_data, overwrite=True)
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_mask{}.fits'.format('_'+ID)),
                     red_data*MASK2MINIMIZE, overwrite=True)
        
        #red_data_cube = np.repeat(red_data[np.newaxis, :, :], len(PA_ARRAY), axis=0)
        #red_data_cube_pa   = cube_derotate(red_data_cube, PA_ARRAY, imlib='opencv')
        #fits.writeto(os.path.join(inputs_resultdir,'reduced_image_cube.fits'),red_data_cube_pa, overwrite=True)

        return red_data
    return None




def save_SNR_map_final():
    '''
    Save the SNR maps.
    Use the function save_one_SNR_map().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        noise_map = NOISE
        SCIENCE =  SCIENCE_DATA
        RED = RED_DATA
        save_one_SNR_map(TYPE_OBS, noise_map, RED, SCIENCE, ID=str(i))
        
    else:
        red_data = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            noise_map = NOISE[i]
            SCIENCE =  SCIENCE_DATA[i]
            RED =  RED_DATA[i]
            save_one_SNR_map(TYPE_OBS, noise_map, RED, SCIENCE, ID=str(i))

    return np.array(red_data)


def save_one_SNR_map(TYPE_OBS, noise_map, RED=None, SCIENCE=None, ID=None):
    '''
    Save one SNR map.
    Use the function save_one_SNR_map().
    '''
    if TYPE_OBS == 'total_intensity':
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_snr{}.fits'.format('_'+ID)), RED/noise_map, overwrite=True)
    else:
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_snr{}.fits'.format('_'+ID)), SCIENCE/noise_map, overwrite=True)
    return
        


def process_REDUCED_DATA_PCA():
    '''
    Return the PCA-reduced data for pupil-tracking reduced observations (assumed to be acquired in 'total_intensity' and not 'polar' light).
    Otherwise, return None.
    Save at the path 'inputs_resultdir' the reduced data in different flavours: only the image, the image masked, or the cube rotating at the different parallactic angles.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .SCIENCE_DATA (2D array if TYPE_OBS = polar, 3D array if TYPE_OBS = total_intensity): 
    .PA_ARRAY (1D array): parallactic angles
    .NMODES (int): number of modes/components to use when applying PCA 
    .MASK_RAD (int): radius of the mask corresponding approximately to the inner working angle of the coronagraph
    .MASK2MINIMIZE (2D array): region where the disk model should match the SCIENCE_DATA. 
    .display (boolean): whether to display information 
    .inputs_resultdir (string): path where to save the files
    '''
    
    if TYPE_OBS == 'total_intensity':
        print('\n(Process the SCIENCE data a second time)')
        print('We reduced the data by applying PCA.')
        print(np.shape(RED_DATA_CUBE_PA))
        red_data =  vip.psfsub.pca_fullfr.pca(RED_DATA_CUBE_PA, PA_ARRAY,
                            ncomp=NMODES, mask_center_px=MASK_RAD,
                            imlib='opencv', full_output=False)

        fits.writeto(os.path.join(inputs_resultdir,'reduced_reduced_image.fits'),
                     red_data, overwrite=True)
        fits.writeto(os.path.join(inputs_resultdir,'reduced_reduced_image_mask.fits'),
                     red_data*MASK2MINIMIZE, overwrite=True)

        
        red_data_cube = np.repeat(red_data[np.newaxis, :, :], len(PA_ARRAY), axis=0)
        red_data_pa   = cube_derotate(red_data_cube, PA_ARRAY, imlib='opencv',  #interpolation='nearneig'
                                )

        fits.writeto(os.path.join(inputs_resultdir,'reduced_reduced_image_cube.fits'),
                     red_data_pa, overwrite=True)

        return None
    return None
    

def apply_MASK2SCIENCE_final():
    '''
    Return the PCA-reduced data for pupil-tracking observation(s).
    Use the function process_one_SCIENCE_PCA().
    '''
    if NOBS == 1: # only one dataset
        TYPE_OBS = TYPE_OBS_ALL
        PA = PA_ARRAY
        SCIENCE =  SCIENCE_DATA
        science_data_mask = apply_one_MASK2SCIENCE(TYPE_OBS, SCIENCE, PA, ID='only')
        
    else:
        science_data_mask = []
        for i in range(NOBS): # loop other the different datasets
            TYPE_OBS = TYPE_OBS_ALL[i]
            PA = PA_ARRAY[i]
            SCIENCE =  SCIENCE_DATA[i]
            science_data_mask.append( apply_one_MASK2SCIENCE(TYPE_OBS, SCIENCE, PA, ID=str(i)) )

    return np.array(science_data_mask)
        

def apply_one_MASK2SCIENCE(TYPE_OBS, SCIENCE, PA, ID=None):
    '''
    Return the science data masked if not in the region of interest. The region of interest is  where the disk model should match the science data.

   # Save at the path 'inputs_resultdir' the masked science data.
    
    Note: assume the following global variables to be defined before calling this function
    .TYPE_OBS (string): set to 'polar' or 'total_intensity' depending on the data
    .SCIENCE_DATA (2D array if TYPE_OBS = polar, 3D array if TYPE_OBS = total_intensity):
    .MASK2MINIMIZE (2D array): region where the disk model should match the SCIENCE_DATA.
    .inputs_resultdir (string): path where to save the files
    
    if TYPE_obs = total_intensity
    .PA_ARRAY (1D array): parallactic angles
    '''
    print('\n(Apply the MASK to the SCIENCE data)')
    if TYPE_OBS == 'polar':
        SCIENCE_DATA_MASK = np.copy(SCIENCE) * MASK2MINIMIZE
        fits.writeto(os.path.join(inputs_resultdir,'reduced_image_mask{}.fits'.format('_'+ID)),
                     SCIENCE_DATA_MASK, overwrite=True)
        
    elif TYPE_OBS == 'total_intensity':
        MASK2MINIMIZE_CUBE = np.repeat(MASK2MINIMIZE[np.newaxis, :, :], len(PA), axis=0)
        MASK2MINIMIZE_CUBE_PA = cube_derotate(MASK2MINIMIZE_CUBE, -PA, imlib='opencv', interpolation='nearneig')
        SCIENCE_DATA_MASK = np.copy(SCIENCE) * MASK2MINIMIZE_CUBE_PA
        fits.writeto(os.path.join(inputs_resultdir,'cube_mask_science{}.fits'.format('_'+ID)), SCIENCE_DATA_MASK, overwrite=True)
        #fits.writeto(os.path.join(inputs_resultdir,'cube_mask{}.fits'.format('_'+ID)), MASK2MINIMIZE_CUBE_PA, overwrite=True)
    return SCIENCE_DATA_MASK


## FINAL TESTS ##

def do_test_disk_empty():
    '''
    Return the value of the chisquare or logarithm of the likelihood (depending if the algorithm of exploration is AMOEBA or MCMC)
    for the absence of disk. This should be worst case scenario, as the disk model then should match the data, resulting in smaller residuals.
    
    Note: assume the following global variables to be defined before:
    .exploration_algo: algorithm of exploration (AMOEBA or MCMC)
    .THETA_INIT (1D array): initial model parameters
    '''
    print('\n- No disk')
    startTime =  datetime.now() 
    theta_init_no_disk = np.copy(THETA_INIT)
    theta_init_no_disk[-1] = 0
    if display: print('Check disk model parameters without flux (scaling_flux should be set to 0):\n', theta_init_no_disk)
        
    if exploration_algo == "MCMC":
        lnpb_model = lnpb(theta_init_no_disk)
        print("Test likelihood on initial model:", lnpb_model)
             
    elif exploration_algo == "AMOEBA":
        chisquare_init_nodisk = chisquare_params(theta_init_no_disk)

    print("Time for a single model: ", datetime.now()  - startTime)
    return chisquare_init_nodisk


def do_test_disk_first_guess():
    '''
    Return the value of the chisquare or logarithm of the likelihood for the first guess of the algorithm of exploration (AMOEBA or MCMC).
    Save files (disk model, disk model convolved to the PSF, best residuals, best residuals normalized by the NOISE).
    
    Note: assume the following global variables to be defined before:
    .exploration_algo: algorithm of exploration (AMOEBA or MCMC)
    .THETA_INIT (1D array): initial model parameters
    .TYPE_OBS (string): observation type (polar or total_intensity)
    .PSF (2D array)
    .SCIENCE_DATA (2D array if TYPE_OBS = polar, 3D array if TYPE_OBS = total_intensity)
    .NOISE (2D array)
    .PA_ARRAY (if TYPE_OBS = total_intensity)
    .MASK2MINIMIZE (2D array): mask within the model should match the SCIENCE_DATA
    .firstguess_resultdir (string): path where to save the files
    '''
    print('\n- Initial model')
    startTime = datetime.now()
    print('Parameter starting point:', THETA_INIT)
        
    if exploration_algo == "MCMC":
        lnpb_model = lnpb(THETA_INIT)
        print("Test likelihood on initial model:", lnpb_model)
        suffix = '_{:.0f}'.format(lnpb_model)
             
    elif exploration_algo == "AMOEBA":
        chisquare_init = chisquare_params(THETA_INIT)
        suffix = '_{:.0f}.'.format(chisquare_init)
        print("chisquare (no-disk) - chisquare (first-guess): {:.0f}".format((chisquare_init_nodisk-chisquare_init)))

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
    
    # Open the parameter file
    if display: print('\nThe configuration file .yaml is:', (str_yalm))
    with open(str_yalm, 'r') as yaml_file:
        params_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    ## Initialize paths
    ## Paths
    DATADIR = params_yaml['DATADIR']
    PREPROCESS_ALL = params_yaml['PREPROCESS']
    POSTPROCESS_ALL = params_yaml['POSTPROCESS']
    FN_PSF_ALL = params_yaml['FN_PSF']
    FN_PA_ALL = params_yaml['FN_PA']
    FN_SCIENCE_ALL = params_yaml['FN_SCIENCE']
    FN_NOISE_ALL = params_yaml['FN_NOISE']

    # Saving directories
    SAVINGDIR = params_yaml['SAVINGDIR'] + date
    os.makedirs(SAVINGDIR,exist_ok=True)
    inputs_resultdir = SAVINGDIR + '/inputs'
    os.makedirs(inputs_resultdir,exist_ok=True)
    firstguess_resultdir = SAVINGDIR + '/first_guess'
    os.makedirs(firstguess_resultdir,exist_ok=True)
    intermediate_resultdir = SAVINGDIR + '/amoeba_intermediate'
    os.makedirs(intermediate_resultdir,exist_ok=True)

    # Log file
    fn_log = "{}/log_diskfit_{}".format(SAVINGDIR,  str_yalm[len('config_files/'):-6] )
    fn_log_info = "{}_info_{}.log".format(fn_log, date)
    sys.stdout = Logger(fn_log_info)
    print("Write a logfile with all printed infos at", fn_log_info)

    if display: print('\nThe configuration file .yaml is:', (str_yalm))
    print('\nSave input files at:\n', inputs_resultdir)
    print('\nSave first guess files at:\n', firstguess_resultdir)

    # Copy yaml file directly in the results folder
    file_destination = inputs_resultdir+'/'
    os.makedirs(file_destination, exist_ok=True)
    print("\nCopy the yaml file as well at the path:\n",file_destination)
    shutil.copy(str_yalm, file_destination)
    
    ## Initialize variables
    # System
    DISTANCE_STAR = params_yaml['DISTANCE_STAR']

    # Observation
    try: NOBS = int(params_yaml['NOBS'])
    except: NOBS = 1
    PIXSCALE_INS = params_yaml['PIXSCALE_INS']
    IBAND = params_yaml['IBAND']
    OWA = params_yaml['OWA'] # OWA = dimension before cropping
    TYPE_OBS_ALL = params_yaml['TYPE_OBS']

    # Processing
    CROP_SCIENCE_ALL =  params_yaml['CROP_SCIENCE']
    CROP_NOISE_ALL =  params_yaml['CROP_NOISE']
    CROP_PSF_ALL =  params_yaml['CROP_PSF']
    NMODES = params_yaml['NMODES']
    #DIMENSION = int(OWA - CROP_SCIENCE*2) # DIMENSION  = dimension after cropping
    MASK_RAD =  params_yaml['MASK_RAD']
    NORM =  params_yaml['SCIENCE_NORM']
    COMBINED =  params_yaml['COMBINED'] 

    # Modelling
    exploration_algo = params_yaml['exploration_algo']
    THETA_INIT = from_param_to_theta_init_8params(params_yaml) # !!!
    NOISE_MULTIPLICATION_FACTOR = params_yaml['NOISE_MULTIPLICATION_FACTOR']
    NPARAMS = int(params_yaml['Nparams_to_fit'])
    PARAMS_TO_FIT = params_yaml['PARAMS_TO_FIT']
    AIN, AOUT = params_yaml['ain_init'], params_yaml['aout_init']
    SAVE_INTERMEDIATE_RESULTS = params_yaml['SAVE_INTERMEDIATE_RESULTS']
    SAVE_DETAIL_RESULTS = params_yaml['SAVE_DETAIL_RESULTS']
    

    if NOBS == 1:
        TYPE_OBS = TYPE_OBS_ALL
        if TYPE_OBS == 'polar':
            DISK_MODEL_POLAR = True
        else:
            DISK_MODEL_POLAR = False
    
    if display:
        print('\n(!) Check the parameters (!)')
        print('\n- System params:')
        print('The star is located at',  DISTANCE_STAR, 'pc.')
        
        print('\n- Observational params:')
        print('The observation is in', TYPE_OBS_ALL, '.')
        if 'total_intensity' in TYPE_OBS_ALL:
            print(NMODES, 'modes are used to apply the PCA.')
            if COMBINED == 0 : print('We consider the channel', IBAND+1, '.')
            else: print('We combined the channels before processing')
        print('The pixel scale is', PIXSCALE_INS, 'as/pix.')
        print('Initially, the field of view is:', OWA*PIXSCALE_INS, 'arsec, i.e., ', OWA, 'pixels.')
        #print('After cropping, the field of view is:', DIMENSION*PIXSCALE_INS, 'arsec, i.e., ', DIMENSION, 'pixels.')
        if NOBS==1: print('The cropping parameters are twice:',CROP_SCIENCE_ALL, '(science),', CROP_PSF_ALL, '(psf), and', CROP_NOISE_ALL, '(noise).')  
        
        print('\n- Processing params:')
        print('The mask radius used is:', MASK_RAD)      
        print('The normalization factor used is:', NORM)
        print('The exploration algo used is:', exploration_algo)
        print('The ain and aout parameters used are:', AIN, AOUT)


    if display:
        print('\n- Load the PSF, (PA_ARRAY), SCIENCE, NOISE, and MASK data ')
    
    ## Load data
    # PSF
    PSF = load_PSF_final()
    
    # PA (if needed, i.e., only for pupil-mode observations)
    PA_ARRAY = load_PA_ARRAY_final()
    
    # Science
    SCIENCE_DATA = load_SCIENCE_final()

    # Noise
    NOISE = load_NOISE_final()
    
    # Define/Load the mask within the residuals will be minimized in the science - model data
    MASK2MINIMIZE = load_MASK2MINIMIZE()
    DIMENSION = int(np.shape(MASK2MINIMIZE)[-1])  # DIMENSION  = dimension after cropping
    NRESEL = float(np.nansum(MASK2MINIMIZE))
    if display: print('Number of pixels considered in the mask is:', NRESEL)

    
    ## Processing
    # Process SCIENCE data (only if we do PCA forward modelling)
    RED_DATA = process_SCIENCE_PCA_final()
    #if TYPE_OBS == 'total_intensity': SCIENCE_DATA = np.copy(RED_DATA_CUBE_PA)

    # Save SNR maps
    save_SNR_map_final()

    #process_REDUCED_DATA_PCA()
        
    # We multiply the SCIENCE data by the mask2minimize
    SCIENCE_DATA_MASK = apply_MASK2SCIENCE_final()


    ## TESTS ## 
    print('\n\n=== Make final test before running the MCMC / AMOEBA exploration algorithm ===')
    detail_resultdir = firstguess_resultdir
    # What is the chisquare if the disk model is empty?
    #chisquare_init_nodisk = do_test_disk_empty()
        
    # What is the chisquare / likelihood of the initial model?
    #chisquare_init = do_test_disk_first_guess()

        
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
        startTime =  datetime.now() 
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
                     datetime.now()  - startTime))

        

    elif exploration_algo == "AMOEBA":
        print('\n=== Start AMOEBA optimization ===')
        startTime =  datetime.now() 
        SAVE_INTERMEDIATE_RESULTS = True
        if SAVE_DETAIL_RESULTS == True: SAVE_DETAIL_RESULTS = False
        bounds = [(10,100), (0,180), (70, 89), (0,1), (0,None), (0,None), (0,None), (0,None)] #params_yaml['bounds'] #!!!
        
        print('The parameters to fit are:', PARAMS_TO_FIT)
        print('(!) The initial guess parameters are:', THETA_INIT)
        print('The bounds are:', bounds)

        print('Bounds:', bounds)
        a = zip(*bounds)
        result_optim = op.minimize(chisquare_params,
                                   THETA_INIT,
                                   method='Nelder-Mead',
                                   #constraints=constr
                                   bounds=bounds,
                                   tol=10
                                   )
        print('Time for the AMOEBA optimization:', datetime.now()  - startTime)
             
        best_theta = result_optim['x']
        chisquare = result_optim['fun']
        niter = result_optim['nfev']

        success  = result_optim['success']
        print('Note:\n-Minimization derived successfully?', success)

        message = result_optim['message']
        print('-Message:', message)

        # Run a last time to save the files
        SAVE_DETAIL_RESULTS = True
        amoeba_final_resultdir = SAVINGDIR +  '/amoeba_final/'
        os.makedirs(amoeba_final_resultdir,exist_ok=True)
        detail_resultdir = amoeba_final_resultdir
        chisquare = chisquare_params(best_theta)
        print('\nSave files at:\n', amoeba_final_resultdir)
        
        print('=== Summary ===')
        try: print('- If there is no disk, \nthe chisquare is %.3e' % chisquare_init_nodisk, 'i.e %.0f' % chisquare_init_nodisk)
        except: pass
        
        try:
            print('- The initial parameters were:\n', THETA_INIT, '\nfor a chisquare = %.3e' % chisquare_init, 'i.e %.0f' % chisquare_init)
            print('- The best parameters derived are:\n', best_theta, '\nfor a chisquare of: %.2e' % chisquare,  'i.e %.0f' % chisquare, 'after', niter, 'iterations.')

        except: pass

        try: 
            diff1 = chisquare_init_nodisk-chisquare_init
            diff2 = chisquare_init-chisquare
            diff3 = chisquare_init_nodisk-chisquare
            print("chisquare (no-disk) - chisquare (first-guess): {:.3e}, i.e., {:.0f}".format(diff1,diff1))
            print("chisquare (first-guess) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff2, diff2))
            print("chisquare (no-disk) - chisquare (best-model): {:.3e}, i.e., {:.0f}.".format(diff3,diff3))
        except: pass


    # copy log file at a second common place
    if 1:
        file_destination =  os.path.join(os.getcwd(),'results/logs/')
        os.makedirs(file_destination, exist_ok=True)
        print("\nCopy the log file as well at the path:\n", file_destination)
        fn_log_info = "log_diskfit_{}_info_{}.log".format(str_yalm[len('config_files/'):-6], date)
        print("logfile = ", fn_log_info)
        shutil.copy(fn_log_info, file_destination+fn_log_info)

     

        