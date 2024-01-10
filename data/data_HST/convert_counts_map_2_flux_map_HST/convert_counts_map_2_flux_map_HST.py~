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

path = 'inputs/*'

flist = glob(path)

for fn in flist:
    im = fits.getdata(fn)

    f = np.copy(im)
    nim = len(np.shape(f))

    ## Derive noise map
    pa_beg, pa_end = 90, 180 # quadrant upper right
    box_xbeg, box_xend = 266, 280 # hide star
    box_ybeg, box_yend = 171, 184

    box_xbeg1, box_xend1 = 240, 248 # hide star
    box_ybeg1, box_yend1 = 228, 237

    box_xbeg2, box_xend2 = 220, 225 # hide star
    box_ybeg2, box_yend2 = 174, 180

    badpixel_x, badpixel_y = 250, 193 # hide bad pixel
    
    # fn = 1 image (not a cube)
    if nim == 2:
        pa_grid = compute_im_pa_grid(f)
        
        # Select the good quadrant in the image
        cond = np.logical_and(pa_grid>pa_beg, pa_grid<pa_end)
        f[~cond]=0
        
        # Hide specific region (e.g. star)
        f[box_ybeg:box_yend, box_xbeg:box_xend]=0
        f[box_ybeg1:box_yend1, box_xbeg1:box_xend1]=0
        f[box_ybeg2:box_yend2, box_xbeg2:box_xend2]=0
        f[badpixel_y,badpixel_x]=0
        
        noise = np.array([compute_limdet_map_ann(f, dr=2, alpha=1, center='n//2', even_or_odd='even', display=0) for i in range(nim)])

    # fn = several images (a cube)
    elif nim > 2:
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
    snr = im/noise

    ## Save files
    namesave = fn[:-5]+'_noise'
    namesave = namesave.replace('inputs','outputs')
    print('File save at:\n', namesave)

    fits.writeto(namesave+'_test_pa_grid.fits', pa_grid, overwrite=True)

    fits.writeto(namesave+'_test_quadrant_used.fits', f, overwrite=True)
    
    fits.writeto(namesave+'.fits', noise, overwrite=True)

    namesave = namesave.replace('noise','snr')
    fits.writeto(namesave+'.fits', snr, overwrite=True)
