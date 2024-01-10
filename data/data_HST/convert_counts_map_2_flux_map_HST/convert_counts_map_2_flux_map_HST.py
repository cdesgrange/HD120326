'''
Script to compute the noise map on a HST image.
Use only one quadrant (to ignore the disk) and remove some regions where are located the stars.
'''

from import_packages_generic import *
from import_functions_generic import *
from functions_derive_noise_map import *
from import_functions_convert_units import *
import vip_hci as vip

from functions_reduction_pca import *
import yaml
import argparse

path = 'inputs/*'

flist = glob(path)

for fn in flist:
    im_counts = fits.getdata(fn)

    ## Convert counts into flux
    # platescale
    platescale = 0.05078 # arcsec
    # inverse sensitivity (from HST data crj main header)
    PHOTFLAM = np.nanmean([4.1488048e-19, 4.1469526e-19]) * u.erg / u.second / u.centimeter**2 / u.AA
    # pivot wavelength
    PHOTPLAM = 5.7435967e+03 * u.AA
    PHOTPHZ = const.c / PHOTPLAM.to(u.m)
    PHOTPHZ = PHOTPHZ.to(u.Hz)

    print('At {:.0f} i.e. {:.2e}: \n-the flux is {:.2e}.'.format(PHOTPLAM, PHOTPHZ, PHOTFLAM))

    # Convert counts into flux ergs/s/cm2
    PHOT_ergs = PHOTFLAM * PHOTPLAM
    print('-the flux is {:.2e}.'.format(PHOT_ergs))

    # Convert flux ergs/s/cm2 into W/m2
    PHOT_Wm2 = PHOT_ergs.to(u.Watt/u.meter**2)

    print('-the flux is {:.2e}.'.format(PHOT_Wm2))

    # Convert flux W/m2 into Jy
    PHOT_Jy = Wm2_to_Jy(PHOT_Wm2, PHOTPHZ) * u.Jy / (u.Watt/u.meter**2/u.Hz)

    print('-the flux is {:.2e}.'.format(PHOT_Jy))

    # Total Exposure Time
    TEXPTIME = np.nanmean([2.046e+3, 2.046e+3])

    
    im_flux = np.copy(im_counts) * PHOT_Jy/TEXPTIME

    im_flux = im_flux.value

    im_flux_platescale = im_flux / platescale**2

    # f = 1 image (not a cube)
 

    ## Save files
    namesave = fn[:-5]+'_flux_jy'
    namesave = namesave.replace('inputs','outputs')
    print('File save at:\n', namesave)
    
    fits.writeto(namesave+'.fits', im_flux, overwrite=True)
    fits.writeto(namesave+'arcsec2.fits', im_flux_platescale, overwrite=True)

