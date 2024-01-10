from import_functions_generic import *
import yaml
import shutil
import vip_hci as vip

L = time.localtime()
date = "{}-{}-{}-{}h{}min{}s".format(L[0],L[1],L[2],L[3],L[4],L[5],L[6])

## FUNCTIONS ##
def HST_mask2coord(mask):
    if mask == 'WedgeA0.6':
        return 309, 69
    elif mask == 'WedgeA1.0':
        return 311, 215
    elif mask == 'WedgeB1.0':
        return 216, 307
    elif mask == 'BAR5':
        return 970, 700
    else:
        raise ValueError('Check how you spell the mask name. It should be either "WedgeA0.6", "WedgeA1.0", "WedgeB1.0" or "BAR5". You indicated:', mask)
    return



## INITIALIZATION ##
fn = 'inputs/aperture.fits'
MASK_TOT = fits.getdata(fn)[0]

display=1
print('\n=== Initialization ===')
str_yaml = sys.argv[1]
    
# Open the parameter file
if display: print('\nThe configuration file .yaml is:', (str_yaml))
with open(str_yaml, 'r') as yaml_file:
    params_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

## Initialize paths
## Paths
#IM = params_yaml['IM']

jobid = select_string_between_characs(str_yaml,'/','.')
SAVING_DIR = 'outputs/' + date + '_'+ jobid +  '/'
os.makedirs(SAVING_DIR,exist_ok=True)

# Log file
fn_log = "{}/log_generate_mask_{}".format(SAVING_DIR,  str_yaml[len('config_files/'):-6] )
fn_log_info = "{}_info_{}.log".format(fn_log, date)
sys.stdout = Logger(fn_log_info)
print("Write a logfile with all printed infos at", fn_log_info)

if display:
    print('\nSave output files at:\n', SAVING_DIR)
  
# Copy yaml file directly in the outputs folder
file_destination = SAVING_DIR
os.makedirs(file_destination, exist_ok=True)
print("\nCopy the yaml file as well at the path:\n",file_destination)
shutil.copy(str_yaml, file_destination)
    
## Initialize variables
# System
ROLLING_ANGLES = params_yaml['ROLLING_ANGLES']

# Observation
PIX_OVERSIZE =  params_yaml['PIX_OVERSIZE']
PIXSCALE = params_yaml['PIXSCALE_INS']
ROWA = params_yaml['ROWA']
MASK_NAME = params_yaml['MASK']
XLOC, YLOC = HST_mask2coord(MASK_NAME)
XLOC, YLOC = int(XLOC), int(YLOC)
#params_yaml['XLOC'],  params_yaml['YLOC']
RAD = ROWA//PIXSCALE
if RAD%2 == 1: RAD = int(RAD+1)
else: RAD = int(RAD)

RSPIDER = params_yaml['RSPIDER']

# Display Parameters
print('\n= Parameters =')
print('The mask used is', MASK_NAME)
print('Scene centered at ({},{}) with size {}x{} pixels^2 i.e. a FOV of {}x{}"^2 '.format(XLOC, YLOC, RAD*2, RAD*2, ROWA*2, ROWA*2))
print('The plate scale = {}"/pixel.'.format(PIXSCALE))
print('The radius  of the spider is  {} pixels.'.format(RSPIDER))


## Compute the occulted regions ##

# Crop
MASK_dum = MASK_TOT[YLOC-RAD:YLOC+RAD, XLOC-RAD:XLOC+RAD]
fits.writeto(SAVING_DIR+'mask.fits', MASK_dum)
MASK = MASK_TOT[YLOC-RAD-PIX_OVERSIZE:YLOC+RAD+PIX_OVERSIZE, XLOC-RAD-PIX_OVERSIZE:XLOC+RAD+PIX_OVERSIZE]

# Spider
n_oversize = 2 * (RAD+PIX_OVERSIZE)
SPIDER1 = np.array( [np.diag([1]*(n_oversize-np.abs(d)),d) for d in range(-RSPIDER,RSPIDER+1) ] )
SPIDER1 = np.sum(SPIDER1, axis=0)
SPIDER = SPIDER1 + SPIDER1[::-1]
SPIDER[SPIDER>1]=1
SPIDER = np.where(SPIDER==1, 0, 1)
fits.writeto(SAVING_DIR+'spiders.fits', SPIDER)


## Apply Mask only
# Rotate Mask
print('Apply the following rolling angles', ROLLING_ANGLES)
CUBE_ROT = []
for rol in ROLLING_ANGLES:
    im_rot = vip.preproc.derotation.frame_rotate(MASK, -rol, mask_val=0,  imlib='opencv', interpolation='nearneig')
    CUBE_ROT.append(im_rot)
CUBE_ROT = np.array(CUBE_ROT)

# Crop
if PIX_OVERSIZE != 0: CUBE_ROT = CUBE_ROT[:,PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]

# Collapse
CUBE_ROT_MASK = np.copy(CUBE_ROT)
fits.writeto(SAVING_DIR+'mask_cube_rot.fits', CUBE_ROT)

# Gradient mask
IM_ROT = np.mean(CUBE_ROT, axis=0)
fits.writeto(SAVING_DIR+'mask_gradient_final.fits', IM_ROT)

# Binary mask: regions systematically masked
IM_OBSCURATOR_MIN = np.copy(IM_ROT)
IM_OBSCURATOR_MIN[IM_OBSCURATOR_MIN != 0] = 1
fits.writeto(SAVING_DIR+'mask_min_binary_final.fits', IM_OBSCURATOR_MIN)

# Binary mask: region sometimes masked
IM_OBSCURATOR_MAX = np.copy(IM_ROT)
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != np.nanmax(IM_OBSCURATOR_MAX)] = 0
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != 0 ] = 1
fits.writeto(SAVING_DIR+'mask_max_binary_final.fits', IM_OBSCURATOR_MAX)


## Apply Spiders only
# Rotate Spiders
print('Apply the following rolling angles', ROLLING_ANGLES)
CUBE_ROT = []
for rol in ROLLING_ANGLES:
    im_rot = vip.preproc.derotation.frame_rotate(SPIDER, -rol, mask_val=0,  imlib='opencv', interpolation='nearneig')
    CUBE_ROT.append(im_rot)
CUBE_ROT = np.array(CUBE_ROT)

# Crop
if PIX_OVERSIZE != 0: CUBE_ROT = CUBE_ROT[:,PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]

# Collapse
CUBE_ROT_SPIDER = np.copy(CUBE_ROT)
fits.writeto(SAVING_DIR+'spiders_cube_rot.fits', CUBE_ROT)

# Gradient mask
IM_ROT = np.mean(CUBE_ROT, axis=0)
fits.writeto(SAVING_DIR+'spiders_gradient_final.fits', IM_ROT)

# Binary mask: regions systematically masked
IM_OBSCURATOR_MIN = np.copy(IM_ROT)
IM_OBSCURATOR_MIN[IM_OBSCURATOR_MIN != 0] = 1
fits.writeto(SAVING_DIR+'spiders_min_binary_final.fits', IM_OBSCURATOR_MIN)

# Binary mask: region sometimes masked
IM_OBSCURATOR_MAX = np.copy(IM_ROT)
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != np.nanmax(IM_OBSCURATOR_MAX)] = 0
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != 0 ] = 1
fits.writeto(SAVING_DIR+'spiders_max_binary_final.fits', IM_OBSCURATOR_MAX)


## Apply both Mask and Spiders
CUBE_ROT = CUBE_ROT_SPIDER * CUBE_ROT_MASK
fits.writeto(SAVING_DIR+'mask+spiders_cube_rot.fits', CUBE_ROT)

# Gradient mask
IM_ROT = np.mean(CUBE_ROT, axis=0)
fits.writeto(SAVING_DIR+'mask+spiders_gradient_final.fits', IM_ROT)

# Binary mask: regions systematically masked
IM_OBSCURATOR_MIN = np.copy(IM_ROT)
IM_OBSCURATOR_MIN[IM_OBSCURATOR_MIN != 0] = 1
fits.writeto(SAVING_DIR+'mask+spiders_min_binary_final.fits', IM_OBSCURATOR_MIN)

# Binary mask: region sometimes masked
IM_OBSCURATOR_MAX = np.copy(IM_ROT)
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != np.nanmax(IM_OBSCURATOR_MAX)] = 0
IM_OBSCURATOR_MAX[IM_OBSCURATOR_MAX != 0 ] = 1
fits.writeto(SAVING_DIR+'mask+spiders_max_binary_final.fits', IM_OBSCURATOR_MAX)






