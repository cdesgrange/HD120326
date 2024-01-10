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

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk
import astro_unit_conversion as convert

os.environ["OMP_NUM_THREADS"] = "1"

from functions_diskfit_mcmc import *

if __name__ == '__main__':
    
    L = time.localtime()
    date = "{}-{}-{}-{}h{}min{}s".format(L[0],L[1],L[2],L[3],L[4],L[5],L[6])
        
    print('\n=== Initialization ===')
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)

    display = 1
    
    if len(sys.argv) == 1:
        #str_yalm = 'SPHERE_Hband_MCMC.yaml'
        str_yalm = 'SPHERE_Hband_AMOEBA_polar_5params.yaml'
    else:
        str_yalm = sys.argv[1]

    if display: print(str_yalm)

    # test on which machine I am

    if display: print(socket.gethostname())
    
    if socket.gethostname() == 'MacBook-Pro-de-sysipag.local':
        #datadir = '/Users/desgranc/Documents/work/projects/HD120326/data/'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        #datadir = '/Users/desgranc/Documents/work/projects/HD120326/data/'
        progress = False

    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)

    if display: print('\nThe configuration file .yaml is located at:\n', yaml_path_file)
    
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    DATADIR = params_mcmc_yaml['DATADIR']
    FN_SCIENCE = params_mcmc_yaml['FN_SCIENCE']
    FN_NOISE = params_mcmc_yaml['FN_NOISE']
    FN_PSF = params_mcmc_yaml['FN_PSF']
    FN_MASK = params_mcmc_yaml['FN_MASK']
    SAVINGDIR = params_mcmc_yaml['SAVINGDIR'] + date
    os.makedirs(SAVINGDIR,exist_ok=True)

    if display:
        print('\nThe path to access data is:\n', DATADIR)
        print('The filename for the reduced image is:\n', FN_SCIENCE)
        print('The filename for the mask is:\n', FN_MASK)
        print('The filename for the PSF is:\n', FN_PSF)
    
    #klipdir = os.path.join(DATADIR, 'klip_fm_files')
    #distutils.dir_util.mkpath(klipdir)

    #mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    #distutils.dir_util.mkpath(mcmcresultdir)

    ## Initialize the things necessary to do a
    #dataset = initialize_mask_psf_noise(params_mcmc_yaml)

    # load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    if display:
        print('\n-Params:')
        print('The star is located at',  DISTANCE_STAR, 'pc.')
        print('The pixel scale is', PIXSCALE_INS, 'as/pix.')

    # DIMENSION = dataset.input.shape[1]
    OWA = params_mcmc_yaml['OWA'] # initially
    cropping =  params_mcmc_yaml['CROPPING']
    DIMENSION = int(OWA - cropping*2) # after cropping
    
    #move_here = params_mcmc_yaml['MOVE_HERE']
    #numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    noise_multiplication_factor = params_mcmc_yaml['NOISE_MULTIPLICATION_FACTOR']

    # The center of the reduced image before cropping
    #xcen = params_mcmc_yaml['xcen']
    #ycen = params_mcmc_yaml['ycen']

    norm2contrast =  params_mcmc_yaml['SCIENCE_NORM']
    if display: print('The normalization factor used is:', norm2contrast)

    # load PSF and make it global
    PSF = fits.getdata(os.path.join(DATADIR, FN_PSF))[0][0]
    PSF /= np.nanmax(PSF)
    
    # load noise and make it global
    NOISE = fits.getdata(os.path.join(DATADIR,FN_NOISE))#[0]  ### we take only the first band 
    #NOISE /= norm2contrast
    NOISE /= np.nanmax(NOISE)
    
    # initialize_diskfm and make diskobj global
    #DISKOBJ = initialize_diskfm(dataset, params_mcmc_yaml)

    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(os.path.join(DATADIR,FN_SCIENCE))#[0]  ### we take only the first band /  KL mode
    #REDUCED_DATA /= norm2contrast
    REDUCED_DATA /= np.nanmax(REDUCED_DATA)
      
    # we multiply the reduced_data by the mask2minimize to avoid having
    # to pass it as a global variable
    mask2minimize = fits.getdata(os.path.join(DATADIR, FN_MASK))
    if display:
        print('\n-Initially')
        print('The size of the reduced image is:', np.shape(REDUCED_DATA))
        print('The size of the noise map is:', np.shape(NOISE))
        print('The size of the mask is:', np.shape(mask2minimize))
        print('The size of  the PSF is:', np.shape(PSF))
        print('The field of view is:', OWA)

    REDUCED_DATA = REDUCED_DATA[cropping:-cropping,cropping:-cropping]
    NOISE = NOISE[cropping:-cropping,cropping:-cropping]
    mask2minimize = mask2minimize[cropping:-cropping,cropping:-cropping]
    
    if display:
        print('\n-After cropping:')
        print('The size of the reduced image is:', np.shape(REDUCED_DATA))
        print('The size of the noise map is:', np.shape(NOISE))
        print('The size of the mask is:', np.shape(mask2minimize))
        print('The size of  the PSF is:', np.shape(PSF))
        print('The dimension of the image is:', DIMENSION)
        
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    REDUCED_DATA *= mask2minimize
    #del mask2minimize, dataset
    params_to_fit = params_mcmc_yaml['params_to_fit']


    # Save files used for forward modelling, MCMC or AMOEBA
    inputs_resultdir = SAVINGDIR + '/inputs'
    os.makedirs(inputs_resultdir,exist_ok=True)
    print('\nSave input files at:\n', inputs_resultdir)

    fits.writeto(os.path.join(inputs_resultdir,'reduced_image_masked.fits'),
                     REDUCED_DATA, overwrite=True)
  
    fits.writeto(os.path.join(inputs_resultdir,'PSF.fits'),
                     PSF, overwrite=True)

    fits.writeto(os.path.join(inputs_resultdir,'NOISE.fits'),
                     NOISE, overwrite=True)
  

     
    # Make a final test by printing the likelihood of the initial model
    if 0:
        print('\n\n=== Make a final test by printing the likelihood of the initial model ===')
        startTime = datetime.now()
        lnpb_model = lnpb(from_param_to_theta_init(params_mcmc_yaml))

        print("Time for a single model: ", datetime.now() - startTime)
        print('Parameter Starting point:', from_param_to_theta_init(params_mcmc_yaml))
        print("Test likelihood on initial model:", lnpb_model)

    exploration_algo = params_mcmc_yaml['exploration_algo']

        
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
        bounds = [(25,50),(80,90), (100, 130), (0,1), (0,None)] #params_mcmc_yaml['bounds']
        
        print('The parameters to fit are:', params_to_fit)
        print('(!) The initial guess parameters are:', theta_init)
        print('The bounds are:', bounds)

        #constr = ({'type': 'ineq', 'fun': lambda x:  x[0] - 100},
        #         )
        print(bounds, type(bounds))
        a = zip(*bounds)
        print(a)
        result_optim = op.minimize(logl_5params,
                                   theta_init,
                                   method='Nelder-Mead',
                                   #constraints=constr
                                   bounds=bounds
                                   )
        print('Time for the AMOEBA optimization:', datetime.now() - startTime)
             
        best_theta = result_optim['x']
        cost = result_optim['fun']
        niter = result_optim['nfev']
        print('(!) The best parameters derived are:', best_theta, 'for a cost of: %.2e' % cost, 'after', niter, 'iterations.')

        success  = result_optim['success']
        print('Note:\n-Minimization derived successfully?', success)

        message = result_optim['message']
        print('-Message:', message)

        amoebaresultdir = SAVINGDIR +  '/results_AMOEBA'
        os.makedirs(amoebaresultdir,exist_ok=True)

        best_model = call_gen_disk_5params(best_theta)
        #best_modelconvolved = convolve(best_model, PSF, boundary='wrap')
        best_modelconvolved = convolve_fft(best_model, PSF, boundary='wrap')
        best_modelconvolved /= np.nanmax(best_modelconvolved)
        #DISKOBJ.update_disk(best_modelconvolved)
        #best_model_fm = DISKOBJ.fm_parallelized()[0]
        #best_residuals = (REDUCED_DATA - best_model_fm)
        best_residuals = (REDUCED_DATA - best_modelconvolved)
        best_residuals_snr = best_residuals / NOISE

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
