from import_packages_generic import *
from import_functions_generic import *
from functions_derive_noise_map import *
import vip_hci as vip

from functions_reduction_pca import *
import yaml
import argparse

path = '/Users/desgranc/Documents/work/projects/HD120326/data/reduced/PCA_Celia/fits/OPP_PA/*IFS*indiv*cube_im_pca.fits'

flist = glob(path)

for fn in flist:
    f = fits.getdata(fn)
    nim = len(f)
    noise_cube = np.array([compute_limdet_map_ann(f[i], dr=2, alpha=1, center='n//2', even_or_odd='even', display=0) for i in range(nim)])

    namesave = fn[:-5]+'_noise.fits'
    namesave = namesave.replace('OPP_PA','NOISE')
    print('File save at:\n', namesave)
    
    fits.writeto(namesave, noise_cube, overwrite=True)
