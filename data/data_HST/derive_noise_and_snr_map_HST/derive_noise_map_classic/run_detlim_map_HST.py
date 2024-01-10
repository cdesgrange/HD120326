"""
Derive 2D detection limit maps by different methods:
- slippy box
- 1D-annulus
- section of annulus
- slippy section of annulus

Compare with SpeCal results

Record info and errors printed in a logfile.

Here: SPHERE IFS data from AU Mic (Gallenne+2022)

Last modification: 2022/03/14, by Célia Desgrange
"""


from import_packages_generic import *
from import_functions_generic import *
from import_functions_detlim_map import *
from matplotlib.colors import LogNorm
from astropy import stats

import shutil

import sys
import logging
import time

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()

L = time.localtime()
date = "{}-{}-{}_{}h{}".format(L[0],L[1],L[2],L[3],L[4],L[5])

sys.stdout = Logger("log_ird_pca_infos_all_{}.txt".format(date))

#error
def log_error(message):
    with open("log_ird_pca_errors_all_{}.txt".format(date), "a+") as myfile:
        myfile.write(f'ERROR : {message} \n')

## Initialization
print('\n===== Run detection limits HST script =====\n')
display=1
print('\n=== Initialization ===')
str_yaml = sys.argv[1]

# Open the parameter file
if display: print('\nThe configuration file .yaml is:', (str_yaml))
with open(str_yaml, 'r') as yaml_file:
    params_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

## Initialize paths
## Paths
jobid = select_string_between_characs(str_yaml,'/','.')
SAVING_DIR = 'outputs/' + date + '_'+ jobid +  '/'
os.makedirs(SAVING_DIR,exist_ok=True)

# Log file
fn_log = "{}/log_derive_detlim_map_{}".format(SAVING_DIR,  str_yaml[len('config_files/'):-6] )
fn_log_info = "{}_info_{}.log".format(fn_log, date)
print('fn_log_info')
sys.stdout = Logger(fn_log_info)
print("Write a logfile with all printed infos at", fn_log_info)

if display:
    print('\nSave output files at:\n', SAVING_DIR)


# Copy yaml file directly in the outputs folder
file_destination = SAVING_DIR
os.makedirs(file_destination, exist_ok=True)
print("\nCopy the yaml file as well at the path:\n",file_destination)
shutil.copy(str_yaml, file_destination)
  

if 1:
    print("deb try")
    # Initialization
    instru = 'HST'
    platescale = params_yaml['PIXSCALE_INS']
    size_box   = params_yaml['SIZE_BOX'] 
    paths = 'inputs/*'

    flist = glob(paths)

    # General parameters
    center, even_or_odd = 'n//2', 'even'
    vmin, vmax = 1e-8, 1e-5

    for k in range(len(flist)) :
        path_fn = flist[k]
        fn = select_string_between_characs(path_fn,'/','.',-1)
        print('\n===== Start to compute the detection limit map for a new epoch =====\n')
        print('filename:',fn)
        print('path_fn:', path_fn)

        label = fn
        res_map = fits.getdata(path_fn)
        print("shape res_map", np.shape(res_map))

        # Plot the positive residual residual map
        res_map_pos = np.copy(res_map)
        #res_map_pos[res_map_pos<0] = 0
        #plot_map(res_map_pos, label+' residual map', vmin = vmin, vmax=vmax)

        # Plot detection limit map - method: slippy box
        dx, dy = size_box, size_box
        fn_detlim = "im_limdet_box_{}_dx={}_dy={}.fits".format(label.replace(' ','_'),dx, dy)
        try : im_limdet_box = fits.getdata(fn_detlim)
        except : im_limdet_box = compute_limdet_im_box_slippy(res_map,dx, dy)
        #plot_map(im_limdet_box, label+' detlim map slippy box', vmin=vmin, vmax=vmax)
        fits.writeto(SAVING_DIR + fn_detlim, im_limdet_box, overwrite=True)

        snrmap = res_map_pos/im_limdet_box
        fits.writeto(SAVING_DIR + 'snrmap.fits', snrmap, overwrite=True)
        

        if 0:
            # Plot detection limit map - method: 1D-annulus
            dr, alpha = 3, 5
            fn_detlim = "im_limdet_ann_{}_dr={}_alpha={}.fits".format(label.replace(' ','_'),dr, alpha)
            try : im_limdet_ann = fits.getdata(fn_detlim)
            except : im_limdet_ann = compute_limdet_map_ann(res_map, dr, alpha=alpha, center=center, even_or_odd=even_or_odd)
            #plot_map(im_limdet_ann, label+' detlim map 1D-annulus', vmin=vmin, vmax=vmax)
            fits.writeto(fn_detlim,  im_limdet_ann, overwrite=True)

            # Plot detection limit map - method: annulus + sections
            #dr, alpha = 3, 5
            nb_pixel_section = 200
            fn_detlim = "im_limdet_ann_sec_{}_dr={}_alpha={}_nbpixels={}.fits".format(label.replace(' ','_'),dr, alpha, nb_pixel_section)
            try : im_limdet_ann_sec = fits.getdata(fn_detlim)
            except : im_limdet_ann_sec = compute_limdet_map_ann_sec(res_map, nb_pixel_section, dr, alpha=alpha, center=center, even_or_odd='even')
            #plot_map(im_limdet_ann_sec, label+' detlim map sections of annulus', vmin=vmin, vmax=vmax)
            # Save im in fits
            fits.writeto(fn_detlim, im_limdet_ann_sec, overwrite=True)

            # Plot detection limit map - method: annulus + slippy sections
            #nb_pixel_section = 200
            fn_detlim = "im_limdet_ann_sec_slippy_{}_dr={}_nbpixels={}.fits".format(label.replace(' ','_'),dr, nb_pixel_section)
            try : im_limdet_ann_sec_slippy = fits.getdata(fn_detlim)
            except : im_limdet_ann_sec_slippy = compute_limdet_map_ann_sec_slippy(res_map, nb_pixel_section, dr,
                center=center, even_or_odd='even', crop_owa = 400)
            #plot_map(im_limdet_ann_sec_slippy, label+' detlim map slippy sections of annulus', vmin=vmin, vmax=vmax)
            # Save im in fits
            fits.writeto(fn_detlim, im_limdet_ann_sec_slippy, overwrite=True)

            # Comparison SpeCal Output
            # Initialization
            infolder    = '/Volumes/antares/work/astro/projects/co-authors/Gallenne2021_AUMic_interfero/charac/limdet_maps/{}/'.format(instru)
            #'/Users/user/Documents/desgrange/work/projects/Gallenne2021_AUMic_interfero/charac/limdet_maps/{}/'.format(instru)
            path_official_detlim = infolder +  epoch + '*'
            print('Path to look for the detection limit map computed by SpeCal:',path_official_detlim)
            fn_detlim = "im_limdet_specal_{}.fits".format(label.replace(' ','_'))
            try :
                try : im_limdet_specal = fits.getdata(fn_detlim)
                except : print(glob(path_official_detlim)) ; im_limdet_specal = fits.getdata( glob(path_official_detlim)[0] )
                #plot_map(im_limdet_specal, label+' detlim map from SpeCal', vmin=vmin, vmax=vmax)
                fits.writeto('im_limdet_specal_'+label+'.fits', im_limdet_specal, overwrite=True)
            except: print('No detection limit map was derived by SpeCal for the epoch {}'.format(epoch))
            # Comparison 1D-contrast curves
            #IM = [im_limdet_box, im_limdet_ann, im_limdet_ann_sec, im_limdet_ann_sec_slippy, im_limdet_specal]
            #LAB = ['slippy box', '1D-annulus', 'section of annulus', 'slippy section of annulus', 'specal']

             #plot_comparison_contrast_curves(IM, LAB, LS=[':','--','-.','-','-.'], pixarc=platescale)

        print("end try")


if 0:
    #else Exception as e:
    log_error(e)
    logging.error("Exception occurred", exc_info=True)
