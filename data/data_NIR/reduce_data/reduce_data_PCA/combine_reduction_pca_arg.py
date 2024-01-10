from import_packages_generic import *
from import_functions_generic import *
import vip_hci as vip

from functions_reduction_pca import *
import yaml
import argparse
import warnings

parser = argparse.ArgumentParser(description='reduction_pca_arg')
parser.add_argument("input", type=str, help='configuration file (.yaml)')
#parser.add_argument("output", type=str, help='dir (will not be created if needed')
args = parser.parse_args()


#### MAIN
## INITIALIZATION TO COMPUTE DETECTION LIMITS
## Parameters to update: yaml file
#fn = 'inputs/IRDIS_2019-06-26_BB_H.yaml'

# read config file
print("\n=== Read the yaml: {}  ===\n".format(args.input))
with open(args.input,"r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


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
print("Number of reductions in total to combine: ",n)

CUBE_IM_PCA, PCS = [], []


## Load each reduction ##
for k in range(n):
    band, epoch, instru, platescale, nmodes = BAND[k], EPOCH[k], INSTRU[k], PLATESCALE[k], NMODES[k]
    label = '{}_{}_{}'.format(epoch,instru,band)
    crop, mask_rad, channels, technique, sortframe = CROP[k], MASK_RAD[k], CHANNELS[k], TECHNIQUE[k], SORTFRAME[k]

    save_fits, plot_modes, plot_res_maps, plot_data = 1, 1, 1, 1
    res_vminmax, res_vmin, res_vmax = 1, 0, 1e-7
    modes_vminmax, modes_vmin, modes_vmax = 1, 0, 1e-3
    platescale, mask_rad = 12.25, 7

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
        pcs  = fits.getdata(saving_dir+'fits/'+namesave+'_cube_pcs'+'.fits')

        CUBE_IM_PCA.append(cube_im_pca)
        PCS.append(pcs)

        print(np.shape(CUBE_IM_PCA[-1]))

    except ImportError:
        msg = "\n!!! The data do not exist, reduced them first with the script run_reduction_pca_arg.py !!!"
        warnings.warn(msg, ImportWarning)


## Combine the data ##
CUBE_IM_PCA = np.array(CUBE_IM_PCA)

cube_im_pca = np.nanmean(CUBE_IM_PCA, axis=0)
pcs = np.nanmean(np.array(PCS), axis=0)
label = '{}epochs_{}_{}'.format(len(CUBE_IM_PCA),instru, np.where(band == 'BB_H', band[-1], band[0]))
namesave = '{}_{}_{}_{}'.format(label, channels, np.where(channels=='both','channels','channel'), technique)

if save_fits and saving_dir != None :
    print("\nWrite the reduced cube by combining different epochs. Different number of modes have been used for the PCA.")
    print("at :", saving_dir+'fits/'+namesave+'.fits')
    fits.writeto(saving_dir+'fits/'+namesave+'_cube_im_pca'+'.fits', cube_im_pca, overwrite=True)
    fits.writeto(saving_dir+'fits/'+namesave+'_cube_pcs'+'.fits', pcs,  overwrite=True)

if plot_data :
    print("Plot")
    plot_pca_modes(pcs, nmodes, saving_dir, namesave=namesave, mask_rad=mask_rad, platescale=platescale,
                vminmax = modes_vminmax, vmin = modes_vmin, vmax = modes_vmax)
    plot_pca_res_map(cube_im_pca, nmodes, saving_dir, namesave=namesave,  mask_rad=mask_rad, platescale=platescale,
                vminmax = res_vminmax, vmin = res_vmin, vmax = res_vmax)
