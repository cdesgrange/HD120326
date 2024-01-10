'''
Script to compute the noise map on a HST image.
Use only one quadrant (to ignore the disk) and remove some regions where are located the stars.
'''

from import_packages_generic import *
from import_functions_generic import *
from functions_derive_noise_map import *
import vip_hci as vip

from functions_reduction_pca import *
import yaml
import argparse

path_signal = 'inputs/*Q_phi*'
path_noise = 'inputs/*U_phi*'

flist = glob(path_noise)
fn_signal = glob(path_signal)[0]

for fn in flist:
    im = fits.getdata(fn)
    im_signal = fits.getdata(fn_signal)

    f = np.copy(im)
    nim = len(np.shape(f))

    ## Derive noise map
    #pa_beg, pa_end = 90, 180 # quadrant upper right
    box_xbeg, box_xend = 714, 731 #266, 280 # hide star
    box_ybeg, box_yend = 586, 600  #171, 184

    box_xbeg1, box_xend1 = 672, 695 #608, 694 # hide star
    box_ybeg1, box_yend1 = 565, 584  #678, 684

    box_xbeg2, box_xend2 = 673, 694 # hide star
    box_ybeg2, box_yend2 = 444, 460

    box_xbeg3, box_xend3 = 717, 732 # hide star
    box_ybeg3, box_yend3 = 434, 443
    
    box_xbeg4, box_xend4 = 453, 466 # hide star
    box_ybeg4, box_yend4 = 398, 413

    box_xbeg5, box_xend5 = 404, 418 # hide star
    box_ybeg5, box_yend5 = 561, 573
    
    #badpixel_x, badpixel_y = 250, 193 # hide bad pixel
    
    # fn = 1 image (not a cube)
    if nim == 2:
        print('The noise file is an image (not a cube).')
        #pa_grid = compute_im_pa_grid(f)
        
        # Select the good quadrant in the image
        #cond = np.logical_and(pa_grid>pa_beg, pa_grid<pa_end)
        #f[~cond]=0
        
        # Hide specific region (e.g. star)
        f[box_ybeg:box_yend, box_xbeg:box_xend]=0
        f[box_ybeg1:box_yend1, box_xbeg1:box_xend1]=0
        f[box_ybeg2:box_yend2, box_xbeg2:box_xend2]=0
        f[box_ybeg3:box_yend3, box_xbeg3:box_xend3]=0
        f[box_ybeg4:box_yend4, box_xbeg4:box_xend4]=0
        f[box_ybeg5:box_yend5, box_xbeg5:box_xend5]=0
        
        #f[badpixel_y,badpixel_x]=0
        noise = np.array(compute_limdet_map_ann(f, dr=2, alpha=0, center='n//2', even_or_odd='even', display=1))

    # fn = several images (a cube)
    elif nim > 2:
        print('The noise file is a cube (not an image).')
        pa_grid = compute_im_pa_grid(f[0])
        
        # Select the good quadrant in the image
        cond = np.logical_and(pa_grid>pa_beg, pa_grid<pa_end)
        f[:, ~cond]=0
        
        # Hide specific region (e.g. star)
        f[:,box_ybeg:box_yend,box_xbeg:box_xend]=0
        f[:,box_ybeg1:box_yend1, box_xbeg1:box_xend1]=0
        f[:,box_ybeg2:box_yend2, box_xbeg2:box_xend2]=0
        
        noise = np.array([compute_limdet_map_ann(f[i], dr=2, alpha=1, center='n//2', even_or_odd='even', display=0) for i in range(nim)])

        
    ## Derive S/N map/cube
    snr = im_signal/noise

    ## Save files
    namesave = fn[:-5]+'_noise'
    namesave = namesave.replace('inputs','outputs')
    print('File save at:\n', namesave)

    #fits.writeto(namesave+'_test_pa_grid.fits', pa_grid, overwrite=True)

    fits.writeto(namesave+'_test_im_noise_used.fits', f, overwrite=True)   
    fits.writeto(namesave+'.fits', noise, overwrite=True)

    namesave = namesave.replace('noise','snr').replace('U_phi','Q_phi')
    fits.writeto(namesave+'.fits', snr, overwrite=True)
