from import_packages_generic import *
from import_functions_generic import *

path = '/Users/desgranc/Documents/work/projects/HD120326/data/reduced/PCA_Celia/fits/NOISE/*IFS*indiv*cube_im_pca*.fits'

flist = glob(path)

for fn in flist:
    fn_noise = fn
    fn_science = fn.replace('NOISE','SCIENCE')
    fn_science = fn_science.replace('_noise','')

    print('\nScience file:\n', fn_science)
    print('Noise file:\n', fn_noise)
    
    f_science = fits.getdata(fn_science)
    f_noise = fits.getdata(fn_noise)

    snr_cube = f_science / f_noise
    snr_cube[np.isnan(snr_cube)]=0
   
    namesave = fn_science[:-5]+'_snr.fits'
    namesave = namesave.replace('SCIENCE','SNR')
    print('File save at:\n', namesave)
    
    fits.writeto(namesave, snr_cube, overwrite=True)
