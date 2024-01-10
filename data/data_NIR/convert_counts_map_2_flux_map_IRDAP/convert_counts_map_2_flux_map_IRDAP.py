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


# function
def ADU2FLUX(flux_adu_obj, flux_adu_star, flux_ref_star, dit_obj, dit_star, transmi_obj, transmi_star, platescale):
    x = flux_adu_star / (dit_star * transmi_star) 
    #print('Star flux = %.3e ADU/s' % x )
    return flux_adu_obj/ flux_adu_star * dit_star/dit_obj * transmi_star/transmi_obj * flux_ref_star 

def ADU2CONTRAST(flux_adu_obj, flux_adu_star, flux_ref_star, dit_obj, dit_star, transmi_obj, transmi_star, platescale):
    x = flux_adu_star / (dit_star * transmi_star) 
    #print('Star flux = %.3e ADU/s' % x )
    return flux_adu_obj/ flux_adu_star * dit_star/dit_obj * transmi_star/transmi_obj

def ADU2CONTRAST_factor(dit_obj, dit_star, transmi_obj, transmi_star, platescale, flux_adu_star):
    return dit_star/dit_obj * transmi_star/transmi_obj * 1/flux_adu_star * 1/platescale

## Params: obs
DIT_STAR, DIT_OBJECT = 0.837464, 64 # s taken from the header of the images
TRANSMISSION_STAR, TRANSMISSION_OBJECT = 0.12591487665785145, 1 # corresponds to ND1.0 and OPEN; taken from the header of the images
PLATESCALE =  12.25e-3 # arcsec/pix 
FLUX_REF_STAR = 0.975 # Jy from Simbad in the filter Johnson H (!!)
flux_adu_star = 723584


for fn in flist:
    im_counts = fits.getdata(fn)

    ## Convert counts into contrast
    factor_adu2contrast = ADU2CONTRAST_factor( DIT_OBJECT, DIT_STAR, TRANSMISSION_OBJECT, TRANSMISSION_STAR, PLATESCALE, flux_adu_star)
    
    print('The factor of conversion ADU to CONTRAST is {:.2e}.'.format(factor_adu2contrast))
    im_contrast_arcsec2 = np.copy(im_counts)*factor_adu2contrast  # contrast/arcsec2

    # Convert contrast into Jy/arcsec2
    im_flux_jy_arcsec2 = np.copy(im_contrast_arcsec2) * FLUX_REF_STAR  # Jy/arcsec2
   
 
    ## Save files
    namesave = fn[:-5]
    namesave = namesave.replace('inputs','outputs')
    print('File save at:\n', namesave)

    fits.writeto(namesave+'_contrast_arcsec2.fits', im_contrast_arcsec2, overwrite=True)
    fits.writeto(namesave+'_flux_jy_arcsec2.fits', im_flux_jy_arcsec2, overwrite=True)


