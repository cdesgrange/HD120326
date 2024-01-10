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
XLOC, YLOC = int(XLOC)-1, int(YLOC)-1
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

## Data ##
PATH_IM_ROLL1 =  params_yaml['PATH_IM_ROLL1']
PATH_IM_ROLL2 =  params_yaml['PATH_IM_ROLL2']
ORIENT1 = params_yaml['ORIENT1']
ORIENT2 = params_yaml['ORIENT2']
drot = ORIENT1-ORIENT2

IM1 = fits.getdata(PATH_IM_ROLL1)
IM2 = fits.getdata(PATH_IM_ROLL2)

interpo_rotate =  'nearneig' #'lanczos4'

print('Shape of the image:', np.shape(IM1))

if np.shape(IM1)[0] != 1024:
    print('The image is bigger than the conventional (1024,1024) shape, do a first crop.')
    crop0 = (np.shape(IM1)[0]-1024)//2
    IM1, IM2 = IM1[crop0:-crop0,crop0:-crop0], IM2[crop0:-crop0,crop0:-crop0]
    print('Now the image shape is',np.shape(IM1))
    
# Crop
IM1 = IM1[YLOC-RAD-PIX_OVERSIZE:YLOC+RAD+PIX_OVERSIZE, XLOC-RAD-PIX_OVERSIZE:XLOC+RAD+PIX_OVERSIZE]
IM2 = IM2[YLOC-RAD-PIX_OVERSIZE:YLOC+RAD+PIX_OVERSIZE, XLOC-RAD-PIX_OVERSIZE:XLOC+RAD+PIX_OVERSIZE]

# Save files
if PIX_OVERSIZE != 0:
    fits.writeto(SAVING_DIR+'IM1_ORIENT1.fits', IM1[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE])
    fits.writeto(SAVING_DIR+'IM2_ORIENT2.fits', IM2[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE])                                           
else:
    fits.writeto(SAVING_DIR+'IM1_ORIENT1.fits', IM1) 
    fits.writeto(SAVING_DIR+'IM2_ORIENT2.fits', IM2)

# Derotation -> nominal angle
IM1_derot = vip.preproc.derotation.frame_rotate(IM1, -ORIENT1, mask_val=0,  imlib='opencv', interpolation=interpo_rotate) # nominal angle
IM2_derot = vip.preproc.derotation.frame_rotate(IM2, -ORIENT2, mask_val=0,  imlib='opencv', interpolation=interpo_rotate) # nominal angle

# Save files
if PIX_OVERSIZE != 0:
    IM1_derot = IM1_derot[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
    IM2_derot = IM2_derot[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
    
fits.writeto(SAVING_DIR+'IM1_derot.fits', IM1_derot)
fits.writeto(SAVING_DIR+'IM2_derot.fits', IM2_derot)
                                              
## Processing
# nADI
IM = np.nanmean([IM1_derot,IM2_derot],axis=0)
IM_model = np.copy(IM)
fits.writeto(SAVING_DIR+'IM_nadi_derot.fits', IM)

# cADI
IM12 = IM1-IM2
IM12_derot = vip.preproc.derotation.frame_rotate(IM12, -ORIENT1, mask_val=0,  imlib='opencv', interpolation=interpo_rotate) # nominal angle
IM21 = -np.copy(IM12)
IM21_derot = vip.preproc.derotation.frame_rotate(IM21, -ORIENT2, mask_val=0,  imlib='opencv', interpolation=interpo_rotate) # nominal angle

# Save files
if PIX_OVERSIZE != 0:
     IM12 = IM12[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
     IM21 = IM21[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
     IM12_derot = IM12_derot[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
     IM21_derot = IM21_derot[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]
     
fits.writeto(SAVING_DIR+'IM12_ORIENT12.fits', IM12)
fits.writeto(SAVING_DIR+'IM21_ORIENT21.fits', IM21)

fits.writeto(SAVING_DIR+'IM12_derot.fits', IM12_derot)
fits.writeto(SAVING_DIR+'IM21_derot.fits', IM21_derot)

IM = np.nanmean([IM12_derot,IM21_derot],axis=0) 
fits.writeto(SAVING_DIR+'IM_cadi_derot.fits', IM)

IM_OBS = np.copy(IM)


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
if PIX_OVERSIZE != 0: fits.writeto(SAVING_DIR+'spiders.fits', SPIDER[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE])
else: fits.writeto(SAVING_DIR+'spiders.fits', SPIDER)

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
for rol in (ROLLING_ANGLES):
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
MS =  MASK*SPIDER
if PIX_OVERSIZE != 0: fits.writeto(SAVING_DIR+'mask+spiders.fits', MS[PIX_OVERSIZE:-PIX_OVERSIZE, PIX_OVERSIZE:-PIX_OVERSIZE]) 
else: fits.writeto(SAVING_DIR+'mask+spiders.fits', MS)

CUBE_ROT = CUBE_ROT_SPIDER * CUBE_ROT_MASK
fits.writeto(SAVING_DIR+'mask+spiders_cube_rot.fits', CUBE_ROT)


# -- Apply the masks to the different reduced data
MS_binary = np.copy(CUBE_ROT[0] )
MS_binary[MS_binary != 0] = 1
IM12_derot_MS_binary = IM12_derot * MS_binary
fits.writeto(SAVING_DIR+'IM12_derot_mask+spiders.fits', IM12_derot_MS_binary)

MS_binary = np.copy(CUBE_ROT[1])
MS_binary[MS_binary != 0] = 1
IM21_derot_MS_binary = IM21_derot * MS_binary
fits.writeto(SAVING_DIR+'IM21_derot_mask+spiders.fits', IM21_derot_MS_binary)
# -- 


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


## Reduction with binary mask applied ##
# Regions systematically masked
IM_OBS_MASK = IM_OBS * IM_OBSCURATOR_MIN
fits.writeto(SAVING_DIR+'im_obs_mask_min_final.fits', IM_OBS_MASK)

# Binary mask: region sometimes masked
IM_OBS_MASK = IM_OBS * IM_OBSCURATOR_MAX
fits.writeto(SAVING_DIR+'im_obs_mask_max_final.fits', IM_OBS_MASK)


## Back to processing: smart cADI

# cADI smart
CUBE_ROT[CUBE_ROT != 0] = 1
im1 = CUBE_ROT[0]*IM12_derot
im2 = CUBE_ROT[1]*IM21_derot
im1_hidden_im2 =  np.where(CUBE_ROT[1]==0, im1, 0)
im2_hidden_im1 =  np.where(CUBE_ROT[0]==0, im2, 0)
im = np.nanmean([im1,im2], axis=0) + im1_hidden_im2 + im2_hidden_im1

fits.writeto(SAVING_DIR+'im_cadi_derot_smart.fits', im)
