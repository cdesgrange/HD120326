from import_packages_generic import *
from import_functions_generic import *
import vip_hci as vip

from functions_reduction_pca import *

import yaml


#### MAIN
## INITIALIZATION TO COMPUTE DETECTION LIMITS
## Parameters to update: yaml file
#fn = 'inputs/IRDIS_2019-06-26_BB_H.yaml'
fn = 'inputs/job3_IRDIS_H23.yaml'
f  = open(fn,"r")
params = yaml.load(f, Loader=yaml.FullLoader)

print("\n=== Read the yaml: {}  ===\n".format(fn))

## General params for the simulations
params_path  = params['paths']
config_reductions  = params['reductions']
data_dir   = params_path['data_dir']
saving_dir = params_path['saving_dir']

## Load the parameters of each obs and band
BAND, EPOCH, INSTRU, PLATESCALE, NMODES = [], [], [], [], []
CROP, MASK_RAD, CHANNELS, TECHNIQUE, SORTFRAME = [], [], [], [], []

for config in config_reductions: # each reduction has its own configuration
    BAND.append(config['band'])
    EPOCH.append(config['epoch'])
    INSTRU.append(config['instru'])
    PLATESCALE.append(config['platescale'])
    NMODES.append(config['nmodes'])
    CROP.append(config['crop'])
    MASK_RAD.append(config['mask_rad'])
    CHANNELS.append(config['channels'])
    TECHNIQUE.append(config['technique'])
    SORTFRAME.append(config['sortframe'])

n = len(config_reductions)
print("Number of reductions in total: ",n)
for k in range(n):
    band, epoch, instru, platescale, nmodes = BAND[k], EPOCH[k], INSTRU[k], PLATESCALE[k], NMODES[k]
    label = '{}_{}_{}'.format(epoch,instru,band)
    crop, mask_rad, channels, technique, sortframe = CROP[k], MASK_RAD[k], CHANNELS[k], TECHNIQUE[k], SORTFRAME[k]

    save_fits, plot_modes, plot_res_maps = 1, 1, 1
    modes_vminmax, modes_vmin, modes_vmax = 0, 0, 1e-2
    res_vminmax, res_vmin, res_vmax = 0, 0, 1e-2

    print("\n=== Consider the epoch {}, instru {} (i.e. platescale {} mas), band {} and the list of NMODES {} ===\n".format(epoch,
                instru, platescale, band, nmodes))
    print("\n=== Will crop the data from {} pixels (in radius) and consider a mask of {} pixels.===\n".format(crop, mask_rad))
    print("The label of the data is {}".format(label))

    namesave = '{}_{}_{}_{}_{}sorting'.format(label, channels, np.where(channels=='both','channels','channel'), technique, sortframe)
    print("The namesave of the data is {}".format(namesave))

    ## load the data reduced if they already exist
    try :
        print("\n(Load the reduced data if they already exist)")
        print("Looked for them at ", saving_dir+'fits/'+namesave+'_cube_im_pca'+'.fits')
        cube_im_pca = fits.getdata(saving_dir+'fits/'+namesave+'_cube_im_pca'+'.fits')
        pcs  = fits.getdata(saving_dir+'fits/'+label+'_cube_pcs'+'.fits')

        print("Plot")
        plot_pca_modes(pcs, nmodes, saving_dir, namesave=label,
                    vminmax = modes_vminmax, vmin = modes_vmin, vmax = modes_vmax)
        plot_pca_res_map(cube_im_pca, nmodes, saving_dir, namesave=label,
                    vminmax = res_vminmax, vmin = res_vmin, vmax = res_vmax)

    except :
        print("\nLoad the data to reduce")
        cube, header, psf, pa, lbd = load_data(data_dir, band, epoch, instru, channels = channels, crop=crop, sortframe=sortframe, saving_dir=saving_dir,
                            label=label)

        #if len(np.shape(cube)) == 3 : cube = np.where( band in ['J2', 'H2', 'K1'],  cube[0], cube[1])

        ## reduce the data by PCA
        # mask step?

        # Compute the PCA residuals map 'im_pca' for different number of components ( to ncomp_max)
        print("\nCompute the PCA reduced residual map")
        if channels == 'both' :
            # Run and collapse pca (in this order or the inverse, depending on the argument technique)
            cube_im_pca, pcs = run_and_collapse_pca(cube, pa, nmodes, psf, technique, mask_rad = mask_rad,  save_fits=save_fits,
                            plot_modes = plot_modes, plot_res_maps = plot_res_maps, saving_dir = saving_dir, namesave = namesave,
                            modes_vminmax = modes_vminmax, modes_vmin = modes_vmin, modes_vmax = modes_vmax,
                            res_vminmax = res_vminmax, res_vmin = res_vmin, res_vmax = res_vmax )


        else : # consider only one channel, so simply run PCA on the data already loaded
            cube_im_pca, pcs = run_pca(cube, pa, nmodes, mask_rad = mask_rad,  save_fits=save_fits,
                            plot_modes = plot_modes, plot_res_maps = plot_res_maps, saving_dir = saving_dir, namesave = namesave,
                            modes_vminmax = modes_vminmax, modes_vmin = modes_vmin, modes_vmax = modes_vmax,
                            res_vminmax = res_vminmax, res_vmin = res_vmin, res_vmax = res_vmax )
