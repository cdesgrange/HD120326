# pylint: disable=C0103
####### This is the MCMC fitting code for fitting a disk #######
import os

import sys

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
from astropy.convolution import convolve

import yaml

import pyklip.instruments.Instrument as Instrument

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm

from emcee import EnsembleSampler
from emcee import backends

# import make_gpi_psf_for_disks as gpidiskpsf

from vip_hci.metrics.scattered_light_disk import ScatteredLightDisk
import astro_unit_conversion as convert

os.environ["OMP_NUM_THREADS"] = "1"


#######################################################
def call_gen_disk(theta):
    """ call the disk model from a set of parameters. 2g SPF
        use DIMENSION, PIXSCALE_INS and DISTANCE_STAR

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
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

    #generate the model

    model = ScatteredLightDisk(nx=DIMENSION,
                               ny=DIMENSION,
                               distance=DISTANCE_STAR,
                               itilt=inc,
                               omega=argperi,
                               pxInArcsec=PIXSCALE_INS,
                               pa=pa,
                               flux_max=norm,
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
                                   'name': 'DoubleHG',
                                   'g': [g1, g2],
                                   'weight': alpha,
                                   'polar': False
                               })
    return model.compute_scattered_light()


########################################################
def logl(theta):
    """ measure the Chisquare (log of the likelihood) of the parameter set.
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

    model = call_gen_disk(theta)

    modelconvolved = convolve(model, PSF, boundary='wrap')
    DISKOBJ.update_disk(modelconvolved)
    model_fm = DISKOBJ.fm_parallelized()[0]

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    res = (REDUCED_DATA - model_fm) / NOISE

    Chisquare = np.nansum(-0.5 * (res * res))

    return Chisquare


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

    if (argperi < 0 or argperi > 180):
        return -np.inf

    if (eccentricity < 0 or eccentricity > 1):
        return -np.inf

    if (ksi0 < 0.1 or ksi0 > 10):  #The aspect ratio
        return -np.inf

    if (g1 < 0.05 or g1 > 0.9999):
        return -np.inf

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
def initialize_mask_psf_noise(params_mcmc_yaml):
    """ initialize the MCMC by preparing the useful things to measure the
    likelyhood (measure the data, the psf the noise, the masks).

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        a dataset a pyklip instance of Instrument.Data
    """

    # if first_time = 1 old the mask, reduced data, noise map, and KL vectors
    # are recalculated. be careful, for some reason the KL vectors are slightly
    # different on different machines. if you see weird stuff in the FM models
    # (for example in plotting the results), just remake them
    first_time = params_mcmc_yaml['FIRST_TIME']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']
    pixscale_ins = params_mcmc_yaml['PIXSCALE_INS']

    owa = params_mcmc_yaml['OWA']
    move_here = params_mcmc_yaml['MOVE_HERE']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]

    noise_multiplication_factor = params_mcmc_yaml[
        'NOISE_MULTIPLICATION_FACTOR']

    klipdir = os.path.join(DATADIR, 'klip_fm_files') + os.path.sep

    #The PSF centers
    xcen = params_mcmc_yaml['xcen']
    ycen = params_mcmc_yaml['ycen']

    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load the PSF and the parangs, crop the data
    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data

    # if params_mcmc_yaml['BAND_DIR'] == 'SPHERE_Hdata':
    #only for SPHERE
    if first_time == 1:
        psf_init = fits.getdata(os.path.join(DATADIR, "psf_sphere_h2.fits"))
        size_init = psf_init.shape[1]
        size_small = 31
        small_psf = psf_init[size_init // 2 - size_small // 2:size_init // 2 +
                             size_small // 2 + 1,
                             size_init // 2 - size_small // 2:size_init // 2 +
                             size_small // 2 + 1]

        small_psf = small_psf / np.max(small_psf)
        small_psf[np.where(small_psf < 0.005)] = 0.

        fits.writeto(os.path.join(DATADIR,
                                  file_prefix + '_PSF_convolution.fits'),
                     small_psf,
                     overwrite='True')

        # load the raw data
        datacube_sphere_init = fits.getdata(
            os.path.join(DATADIR, "cube_H2.fits")
        )  ### we divide the data to keep the ~same prior as is GPI
        parangs = fits.getdata(os.path.join(DATADIR, "parang.fits"))
        parangs = parangs - 135.99 + 90  ## true north Maire et al. 2016

        datacube_sphere_init = np.delete(datacube_sphere_init, (72, 81),
                                         0)  ## 2 slices are bad
        parangs = np.delete(parangs, (72, 81), 0)  ## 2 slices are bad

        datacube_sphere_init = datacube_sphere_init[
            0:20]  ## we keep only 20 slides to run test on my laptop
        parangs = parangs[
            0:20]  ## we keep only 20 slides to run test on my laptop

        olddim = datacube_sphere_init.shape[1]

        # we resize the SPHERE data to the same size as GPI (281)
        # to avoid a problem of centering
        newdim = 281
        datacube_sphere_newdim = np.zeros(
            (datacube_sphere_init.shape[0], newdim, newdim))

        for i in range(datacube_sphere_init.shape[0]):
            datacube_sphere_newdim[i, :, :] = datacube_sphere_init[
                i, olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1,
                olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1]

        # we flip the dataset (and therefore inverse the parangs) to obtain
        # the good PA after pyklip reduction
        parangs = -parangs

        for i in range(datacube_sphere_newdim.shape[0]):
            datacube_sphere_newdim[i] = np.flip(datacube_sphere_newdim[i],
                                                axis=0)

        datacube_sphere = datacube_sphere_newdim

        fits.writeto(os.path.join(DATADIR, file_prefix + '_true_parangs.fits'),
                     parangs,
                     overwrite='True')

        fits.writeto(os.path.join(DATADIR, file_prefix + '_true_dataset.fits'),
                     datacube_sphere,
                     overwrite='True')

    datacube_sphere = fits.getdata(
        os.path.join(DATADIR, file_prefix + '_true_dataset.fits'))
    parangs_sphere = fits.getdata(
        os.path.join(DATADIR, file_prefix + '_true_parangs.fits'))

    size_datacube = datacube_sphere.shape
    centers_sphere = np.zeros((size_datacube[0], 2)) + [xcen, ycen]
    dataset = Instrument.GenericData(datacube_sphere,
                                     centers_sphere,
                                     parangs=parangs_sphere,
                                     flipx=True,
                                     wvs=None)

    # else:
    #     #only for GPI
    #     if first_time == 1:

    #         filelist4psf = sorted(
    #             glob.glob(os.path.join(DATADIR, "*_distorcorr.fits")))

    #         dataset4psf = GPI.GPIData(filelist4psf, quiet=True)

    #         # identify angles where the
    #         # disk intersect the satspots
    #         excluded_files = gpidiskpsf.check_satspots_disk_intersection(
    #             dataset4psf, params_mcmc_yaml, quiet=True)

    #         # exclude those angles for the PSF measurement
    #         for excluded_filesi in excluded_files:
    #             if excluded_filesi in filelist4psf:
    #                 filelist4psf.remove(excluded_filesi)

    #         # create the data this time wihtout the bad files
    #         dataset4psf = GPI.GPIData(filelist4psf, quiet=True)

    #         # Find the IFS slices for which the satspots are too faint
    #         # if SNR time_mean(sat spot) <3 they are removed
    #         # Mostly for K2 and sometime K1
    #         excluded_slices = gpidiskpsf.check_satspots_snr(dataset4psf,
    #                                                         params_mcmc_yaml,
    #                                                         quiet=True)

    #         params_mcmc_yaml['EXCLUDED_SLICES'] = excluded_slices

    #         # extract the data this time wihtout the bad files nor slices
    #         dataset4psf = GPI.GPIData(filelist4psf,
    #                                   quiet=True,
    #                                   skipslices=excluded_slices)

    #         # finally measure the good psf
    #         instrument_psf = gpidiskpsf.make_collapsed_psf(dataset4psf,
    #                                                        params_mcmc_yaml,
    #                                                        boxrad=15)

    #         # save the excluded_slices in the psf header
    #         hdr_psf = fits.Header()
    #         hdr_psf['N_BADSLI'] = len(excluded_slices)
    #         for badslice_i, excluded_slices_num in enumerate(excluded_slices):
    #             hdr_psf['BADSLI' +
    #                     str(badslice_i).zfill(2)] = excluded_slices_num

    #         #save the psf
    #         fits.writeto(os.path.join(DATADIR,
    #                                   file_prefix + '_SatSpotPSF.fits'),
    #                      instrument_psf,
    #                      header=hdr_psf,
    #                      overwrite=True)

    #     filelist = sorted(glob.glob(os.path.join(DATADIR,
    #                                              "*_distorcorr.fits")))

    #     # in the general case we can choose to
    #     # keep the files where the disk intersect the disk.
    #     # We can removed those if rm_file_disk_cross_satspots == 1
    #     rm_file_disk_cross_satspots = params_mcmc_yaml[
    #         'RM_FILE_DISK_CROSS_SATSPOTS']
    #     if rm_file_disk_cross_satspots == 1:
    #         dataset_for_exclusion = GPI.GPIData(filelist, quiet=True)
    #         excluded_files = gpidiskpsf.check_satspots_disk_intersection(
    #             dataset_for_exclusion, params_mcmc_yaml, quiet=True)
    #         for excluded_filesi in excluded_files:
    #             if excluded_filesi in filelist:
    #                 filelist.remove(excluded_filesi)

    #     # load the bad slices in the psf header
    #     hdr_psf = fits.getheader(
    #         os.path.join(DATADIR, file_prefix + '_SatSpotPSF.fits'))

    #     excluded_slices = []
    #     if hdr_psf['N_BADSLI'] > 0:
    #         for badslice_i in range(hdr_psf['N_BADSLI']):
    #             excluded_slices.append(hdr_psf['BADSLI' +
    #                                            str(badslice_i).zfill(2)])

    #     # load the raw data without the bad slices
    #     dataset = GPI.GPIData(filelist, quiet=True, skipslices=excluded_slices)

    #     #collapse the data spectrally
    #     dataset.spectral_collapse(align_frames=True, numthreads=1)

    #define the outer working angle
    dataset.OWA = owa

    #assuming square data
    dimension = dataset.input.shape[2]

    #create the masks
    if first_time == 1:
        # not use for Grater disk modeling I think.

        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is tightely adjusted to the expected models BEFORE convolution
        # mask_disk_zeros = make_disk_mask(
        #     dimension,
        #     params_mcmc_yaml['pa_init'],
        #     params_mcmc_yaml['inc_init'],
        #     convert.au_to_pix(45, pixscale_ins, distance_star),
        #     convert.au_to_pix(105, pixscale_ins, distance_star),
        #     xcen=xcen,
        #     ycen=ycen)
        # mask2generatedisk = 1 - mask_disk_zeros
        # fits.writeto(os.path.join(klipdir,
        #                           file_prefix + '_mask2generatedisk.fits'),
        #              mask2generatedisk,
        #              overwrite='True')

        # we create a second mask for the minimization a little bit larger
        # (because model expect to grow with the PSF convolution and the FM)
        # and we can also exclude the center region where there are too much speckles
        mask_disk_zeros = make_disk_mask(
            dimension,
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(40, pixscale_ins, distance_star),
            convert.au_to_pix(130, pixscale_ins, distance_star),
            xcen=xcen,
            ycen=ycen)

        mask_speckle_region = np.ones((dimension, dimension))
        # x = np.arange(dimension, dtype=np.float)[None,:] - xcen
        # y = np.arange(dimension, dtype=np.float)[:,None] - ycen
        # rho2d = np.sqrt(x**2 + y**2)
        # mask_speckle_region[np.where(rho2d < 21)] = 0.
        mask2minimize = mask_speckle_region * (1 - mask_disk_zeros)

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2minimize.fits'),
                     mask2minimize,
                     overwrite='True')

    # mask2generatedisk = fits.getdata(
    #     os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))
    mask2minimize = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2minimize.fits'))

    if first_time == 1:
        #measure the noise by reducing with random parangs
        previous_PAs = np.copy(dataset.PAs)
        dataset.PAs = np.random.uniform(0, 360, size=(len(dataset.PAs)))
        parallelized.klip_dataset(dataset,
                                  numbasis=numbasis,
                                  maxnumbasis=120,
                                  annuli=1,
                                  subsections=1,
                                  mode='ADI',
                                  outputdir=klipdir,
                                  fileprefix=file_prefix + '_randomParangs',
                                  aligned_center=[xcen, ycen],
                                  highpass=False,
                                  minrot=move_here,
                                  calibrate_flux=False,
                                  verbose=False)

        reduced_data_wahhajtrick = fits.getdata(
            os.path.join(klipdir,
                         file_prefix + '_randomParangs-KLmodes-all.fits'))[0]
        noise = make_noise_map_no_mask(reduced_data_wahhajtrick,
                                       xcen=xcen,
                                       ycen=ycen,
                                       delta_raddii=3)
        noise[np.where(noise == 0)] = np.nan

        #### We know our noise is too small
        noise = noise_multiplication_factor * noise

        fits.writeto(os.path.join(klipdir, file_prefix + '_noisemap.fits'),
                     noise,
                     overwrite='True')

        dataset.PAs = previous_PAs
        # os.remove(
        #     os.path.join(klipdir,
        #                  file_prefix + '_randomParangs-KLmodes-all.fits'))
        del reduced_data_wahhajtrick

    return dataset


########################################################
def initialize_diskfm(dataset, params_mcmc_yaml):
    """ initialize the MCMC by preparing the diskFM object

    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        a  diskFM object
    """

    first_time = params_mcmc_yaml['FIRST_TIME']
    xcen = params_mcmc_yaml['xcen']
    ycen = params_mcmc_yaml['ycen']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    klipdir = os.path.join(DATADIR, 'klip_fm_files')

    if first_time == 1:
        # create a first model to check the begining parameter and initialize the FM.
        # We will clear all useless variables befire starting the MCMC
        # Be careful that this model is close to what you think is the minimum
        # because the FM is not completely linear so you have to measure the FM on
        # something already close to the best one

        theta_init = from_param_to_theta_init(params_mcmc_yaml)

        #generate the model
        model_here = call_gen_disk(theta_init)

        fits.writeto(os.path.join(klipdir, file_prefix + '_FirstModel.fits'),
                     model_here,
                     overwrite='True')

        model_here_convolved = convolve(model_here, PSF, boundary='wrap')
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel_Conv.fits'),
                     model_here_convolved,
                     overwrite='True')

    model_here_convolved = fits.getdata(
        os.path.join(klipdir, file_prefix + '_FirstModel_Conv.fits'))

    if first_time == 1:
        # initialize the DiskFM object
        diskobj = DiskFM(dataset.input.shape,
                         numbasis,
                         dataset,
                         model_here_convolved,
                         basis_filename=os.path.join(
                             klipdir, file_prefix + '_klbasis.h5'),
                         save_basis=True,
                         aligned_center=[xcen, ycen])
        # measure the KL basis and save it
        fm.klip_dataset(dataset,
                        diskobj,
                        numbasis=numbasis,
                        maxnumbasis=120,
                        annuli=1,
                        subsections=1,
                        mode='ADI',
                        outputdir=klipdir,
                        fileprefix=file_prefix,
                        aligned_center=[xcen, ycen],
                        mute_progression=True,
                        highpass=False,
                        minrot=move_here,
                        calibrate_flux=False)

    # load the the KL basis and define the diskFM object
    diskobj = DiskFM(dataset.input.shape,
                     numbasis,
                     dataset,
                     model_here_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)

    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    modelfm_here = diskobj.fm_parallelized()[
        0]  ### we take only the first KL modemode
    fits.writeto(os.path.join(klipdir, file_prefix + '_FirstModel_FM.fits'),
                 modelfm_here,
                 overwrite='True')

    ## We have initialized the variables we need and we now cleaned the ones that do not
    ## need to be passed to the cores during the MCMC
    return diskobj


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
    """ create a initial set of MCMCparameter from the initial parmeters
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

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)

    if len(sys.argv) == 1:
        str_yalm = 'SPHERE_Hband_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    # test on which machine I am
    # @Sophia you need to adjust that on your own machines
    if socket.gethostname() == 'macbookajm':
        basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/tycho/'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        basedir = '/home/jmazoyer/data_python/tycho/'
        progress = False

    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    klipdir = os.path.join(DATADIR, 'klip_fm_files')
    distutils.dir_util.mkpath(klipdir)

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    distutils.dir_util.mkpath(mcmcresultdir)

    # initialize the things necessary to do a
    dataset = initialize_mask_psf_noise(params_mcmc_yaml)

    # load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    DIMENSION = dataset.input.shape[1]

    # load PSF and make it global
    PSF = fits.getdata(
        os.path.join(DATADIR, FILE_PREFIX + '_PSF_convolution.fits'))

    # load wheremask2generatedisk and make it global
    # WHEREMASK2GENERATEDISK = (fits.getdata(
    #     os.path.join(klipdir, FILE_PREFIX + '_mask2generatedisk.fits')) == 0)

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(klipdir, FILE_PREFIX + '_noisemap.fits'))

    # initialize_diskfm and make diskobj global
    DISKOBJ = initialize_diskfm(dataset, params_mcmc_yaml)

    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode

    # we multiply the reduced_data by the mask2minimize to avoid having
    # to pass it as a global variable
    mask2minimize = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '_mask2minimize.fits'))
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    REDUCED_DATA *= mask2minimize
    del mask2minimize, dataset
    print("")
    print("")
    # Make a final test by printing the likelihood of the initial model
    startTime = datetime.now()
    lnpb_model = lnpb(from_param_to_theta_init(params_mcmc_yaml))

    print("Time for a single model: ", datetime.now() - startTime)
    print('Parameter Starting point:',
          from_param_to_theta_init(params_mcmc_yaml))
    print("Test likelihood on initial model:", lnpb_model)

    exploration_algo = params_mcmc_yaml['exploration_algo']

    if exploration_algo == "MCMC":
        print("Initialization MCMC")
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
        ## Celia: what is AMOEBA?

        print("Start AMOEBA optimization")
        result_optim = op.minimize(logl,
                                   from_param_to_theta_init(params_mcmc_yaml),
                                   method='Nelder-Mead')
        best_theta = result_optim['x']

        amoebaresultdir = os.path.join(DATADIR, 'results_AMOEBA')

        best_model = call_gen_disk(best_theta)
        best_modelconvolved = convolve(best_model, PSF, boundary='wrap')
        DISKOBJ.update_disk(best_modelconvolved)
        best_model_fm = DISKOBJ.fm_parallelized()[0]
        best_residuals = (REDUCED_DATA - best_model_fm)
        best_residuals_snr = best_residuals / NOISE

        fits.writeto(amoebaresultdir.joinpath('best_model_amoeba.fits'),
                     best_model,
                     overwrite=True)
        fits.writeto(
            amoebaresultdir.joinpath('best_modelconvolved_amoeba.fits'),
            best_modelconvolved,
            overwrite=True)
        fits.writeto(amoebaresultdir.joinpath('best_model_fm_amoeba.fits'),
                     best_model_fm,
                     overwrite=True)
        fits.writeto(amoebaresultdir.joinpath('best_residuals_amoeba.fits'),
                     best_residuals,
                     overwrite=True)
        fits.writeto(
            amoebaresultdir.joinpath('best_residuals_snr_amoeba.fits'),
            best_residuals_snr,
            overwrite=True)
