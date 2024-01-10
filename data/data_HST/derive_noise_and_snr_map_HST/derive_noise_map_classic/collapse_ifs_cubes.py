
from import_packages_generic import *
from import_functions_generic import *
from import_functions_detlim_map import *
from matplotlib.colors import LogNorm
from astropy import stats

def wavelength2filter(w):
    '''
    The wavelength should be given in microns
    '''
    if w < 1.1 : return 'Y'
    elif w < 1.4 : return 'J'
    else : return 'H'

def list_wavelength2filter(W):
    '''
    The wavelength should be given in microns
    '''
    BAND = np.array([ wavelength2filter(W[i]) for i in range(len(W))])
    return BAND

# Initialization
infolder    = '/Volumes/antares/work/astro/projects/co-authors/Gallenne2021_AUMic_interfero/'
#'/Users/user/Documents/desgrange/work/projects/Gallenne2021_AUMic_interfero/'
folder_data = 'data/reduced/SPHERE/SPHERE_DC_DATA/'
algo = 'PCA'
convert = 'convert'
folder_reduction = '/*ifs*/'
folder_images = 'Reduction_0000/Images/allcomb/'
fn = "cube_reduced_image_median_corrthput.fits"

paths = infolder + folder_data + algo + folder_reduction + folder_images + fn
print("paths", paths)
flist = glob(paths)

# General parameters
center, even_or_odd = 'n//2', 'even'
vmin, vmax = 1e-8, 1e-5

for k in range(len(flist)) :
    path_fn = flist[k]
    before_fn = select_string_before_charac(path_fn,'/',-1)
    fn = select_string_between_characs(path_fn,'/','/',-5)
    print('\n===== Start to collapse the IFS datacube for a new epoch =====\n')
    print('filename:',fn)
    print('path_fn:', path_fn)

    filt  = select_string_between_characs(fn,'_','_',1)
    epoch = select_string_between_characs(fn,'_','_',2)
    label = '{} {} {}'.format(epoch, filt, algo)
    cube = fits.getdata(path_fn)[1:]
    print("shape cube", np.shape(cube))
    print("\n",infolder+folder_data+convert+'/*{}*ifs*convert*/*lam*'.format(epoch))
    print(glob(infolder+folder_data+convert+'/*{}*ifs*convert/*lam*'),"\n")

    ## Collapse
    W = fits.getdata(glob(infolder+folder_data+convert+'/*{}*ifs*convert*/*lam*'.format(epoch))[0])[1:]
    print("Wavelengths:", W)
    BANDS = list_wavelength2filter(W)
    print("Corresponding bands:", BANDS)

    L = ['Y', 'J', 'H']

    for k in range(len(L)):
        band = L[k]
        # collapse spectral channel for one given band
        cond = BANDS==band
        im = np.median(cube[cond], axis=0)

        if np.any(cond) :
            namesave = "/reduced_image_median_corrthput_band_{}.fits".format(band)
            print("Saving image at", before_fn+namesave)
            fits.writeto(before_fn+namesave, im, overwrite=True)
