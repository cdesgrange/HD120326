# Version 2023-05-23
# Old way to extract flux within an aperture
# Next version: I will use the astropy function, more accurate to extract the flux within a circle and not a square (in particular for small apertures)

print('=== Journal to derive spectra of the disk HD120326 ===')

## Block 0: STANDARD INITIALIZATION ##

## Import packages, functions
import glob
import numpy as np
import time, os
from import_functions_generic import * 
from import_functions_plotting import * 

from functions_background import *
from functions_derive_noise_map import *

# Update Figure outline
plt.style.use('classic')  
mpl.rc('image', cmap='viridis', origin='lower')

rcParams.update({'font.size': 12,  'axes.labelsize' : 12, 'legend.fontsize' : 11,
                 "font.family": "serif", 'text.usetex' : True, "font.serif": [], "font.sans-serif": [], 'mathtext.fontset':"stix",
'legend.handlelength': 1.5, 'legend.borderaxespad' : 0.8, 'legend.columnspacing' : 1, 'legend.handletextpad' : 0.5})

# cmaps
cmap_dark_red =  seaborn.blend_palette([[1,0,0],'crimson','firebrick','darkred',[0.3,0,0]], as_cmap=True)
cmap_red = seaborn.blend_palette(['white',[1,0.95,0],'orange', 'red'], as_cmap=True)
cmap_orange = seaborn.blend_palette(['white',[1,0.95,0],'gold'], as_cmap=True)
cmap_pinkish = seaborn.blend_palette(['white',[1,0.5,1],'purple', 'blue'], as_cmap=True)
cmap_pink = seaborn.blend_palette(['white',[1,0.4,1],[0.8,0,0.7],[0.75,0,0.7]], as_cmap=True)
cmap_blue = seaborn.blend_palette(['white',[0.5,0.9,1], [0,0.5,1],  [0,0,1]], as_cmap=True)
cmap_blue_light = seaborn.blend_palette(['white',[0.5,0.9,1], [0,0.5,1]], as_cmap=True)
cmap_white = seaborn.blend_palette(['white', 'ivory'], as_cmap=True)
cmap = cmap_white

list_rainbow = ['blue','skyblue','hotpink', 'purple','gold','red','orange']
colors_rainbow = seaborn.blend_palette(list_rainbow,7)
#display(colors_rainbow)

MARKERS_IFS = ['<','>','<','>','D','D','o']
COLORS_IFS = colors_rainbow


## Initialize saving directory
L = time.localtime()
date = "{}-{}-{}".format(L[0],L[1],L[2],L[3],L[4],L[5])

saving_dir = 'figs/fig_v{}/'.format(date)
os.makedirs(saving_dir,exist_ok=True)


## Functions

def synthetic_spectrum_star2wgrid_obs(fn_syn, w_obs, plot=0, saving_dir='', show=1):
    '''
    Return the synthetic spectrum (nominal file "fn_syn") on the wavelength grid
    of the observation ("fn_wgrid_obs").
    '''
    ## Load obs
    # w_obs = fits.getdata(fn_wgrid_ifs)
    
    # Load synthethic file
    file = np.loadtxt(fn_syn)
    w, f = file[:,0], file[:,1]
    
    # Add units
    w_units = w * u.Angstrom
    f_units = f * units.erg / u.s / u.cm**2 / u.Angstrom

    # Remove the wavelength in the flux
    f_units *= w_units

    # Convert
    f_units = f_units.to(u.W / u.m **2)
    w_units = w_units.to(u.micron)

    w, f = w_units.value, f_units.value

    # Restrict the region of interest
    cond_w = np.logical_and(w > 0.9, w < 1.6)

    X, Y = w[cond_w], f[cond_w]

    # Remove high frequencies
    Y_BF = savgol_filter(Y,window_length=2000, polyorder=3, mode='nearest')

    # Do it on the wavelength grid of the observation
    Y_OBS = griddata(X, Y_BF, w_obs)
    X_OBS = w_obs

    if plot:
        # Figure
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        ax.semilogy(X, Y, label='nominal synthetic spectrum', color = 'dodgerblue')
        ax.semilogy(X, Y_BF, label='low-filtered synthethic spectrum', color='navy', lw=2)
        ax.semilogy(X_OBS, Y_OBS, label='OBS', color='crimson', lw=0, marker='o')
        ax.set_xlabel('Wavelength ($\mu$m)')
        ax.set_ylabel('Flux (W/m$^2$)')
        ax.legend(loc=0, numpoints=1)
        mise_en_page(ax)
        plt.tight_layout()
        plt.savefig(saving_dir+'synthetic_spectrum.pdf')
        if show: plt.show()
        plt.clf()
    return Y_OBS


## Block 1: INITIALIZATION FOR THIS SYSTEM ##

# Read config file
parser = argparse.ArgumentParser(description='derive spectrum')
parser.add_argument("config", type=str, help='configuration file')
args = parser.parse_args()
print("\nRead the yaml file: {}\n".format(args.config))
with open(args.config,"r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

## Initialize the path to the wished data 
# (!) -> supposed to be changed accordingly to the data/computer used
params_path  = params['paths'] 
infolder = params_path['infolder'] 
folder_prereduced =  params_path['folder_prereduced'] 
folder_reduced = params_path['folder_reduced'] 
dF       = pd.read_csv(infolder + 'list_obs.txt',sep='\t')

print('The epochs of observations are:')
print(dF)


## Parameters
params_plot =  params['plot']
writeto = True # save .fits file
scalebar_pix, scalebar_leg = 100, '0.1"'
save_png = 0    # save .png file
plot_show =  params_plot['show']

# Image
add_colorbar=0; cbar_xdeb = 0.83
figsize_rectangle, figsize_square  = (4,3), (3,3)
figsize = np.where(add_colorbar, figsize_rectangle, figsize_square)
left, right = 0.01, float(np.where(add_colorbar, 0.78, 0.99))
top, bottom = float(np.where(add_colorbar, 0.95, 0.99)), float(np.where(add_colorbar, 0.05, 0.01))
text_xfrac, text_yfrac = 0.03, 0.84

##
im_crop_pca = 8 #9 #im_crop_PCA
im_crop_mask = 8


## Mask
# Large mask IFS
folder_mask = params_path['folder_mask'] 
fn_mask = params_path['fn_mask'] 
path_mask = folder_mask + fn_mask
print('\nLoad the IFS mask located at:\n', path_mask)
im_mask_ifs = fits.getdata(path_mask)
platescale = 7.46
plot_mask =  params_plot['plot_mask']

# Display it
namesave = 'im_{}'.format(fn_mask.replace('.fits',''))
im = np.copy(im_mask_ifs)
im_crop = im_crop_mask

print('-> Size of the image =', len(im), 'and crop Mask =', im_crop, 'i.e. size of the image shown =',  
      len(im_mask_ifs)-2*im_crop)

clab = 'contrast'
vmin, vmax = None, None
text = ''#'{} {}\n{}'.format(epoch, filt, algo)

if plot_mask:
    plot_fig(im=im, vmin=vmin, vmax=vmax, im_crop=im_crop, figsize=figsize,
         scalebar_pix=scalebar_pix, scalebar_leg=scalebar_leg, platescale=platescale,
         add_colorbar=add_colorbar, clab=clab, cbar_xdeb=cbar_xdeb, 
         right=right, left=left, top=top, bottom=bottom,
         text=text, text_xfrac=text_xfrac, text_yfrac=text_yfrac,
         saving_dir=saving_dir, save_png=save_png, namesave=namesave)


## BLOCK 3: LOAD ONE SPECIFIC EPOCH ##
## Load the files to derive the spectrum of the object
params_config = params['config'] 
idx, before_or_after, sorting,channel =  params_config['index'],  '*', 'sorting','*'
epoch = str(dF['epoch'].iloc[idx]); instru=str(dF['instrument'].iloc[idx])
filt =  str(dF['filter'].iloc[idx])
if filt == 'BB_H': filt = 'BBH'
if instru == 'IFS': platescale=7.46
print('Load the epoch: {}, instru: {}, filter: {}'.format(epoch,instru,filt))

algo = 'PCA'

# Opposite PA
print('\nLoad file opposite parallactic angles variation (to derive error bars)')
algo_folder = algo + '_Celia/fits/OPP_PA/'
path_im_reduced = infolder + folder_reduced + algo_folder + '*{}*{}*indiv*{}*{}*{}_cube_im_pca.fits'.format(epoch, instru,
                before_or_after,sorting,channel)
        
#print('It corresponds do the path:\n{}'.format(path_im_reduced))
print('Files found:')
flist = np.sort(glob(path_im_reduced))
for f in flist: print('-',f)

CUBE_REDUCED_OPP_PA = np.array( [ fits.getdata(flist[i]) for i in range(len(flist)) ] )

# Good PA
print('\nLoad file good parallactic angles variation')
algo_folder = algo + '_Celia/fits/SCIENCE/'
path_im_reduced = infolder + folder_reduced + algo_folder + '*{}*{}*indiv*{}*{}*{}_cube_im_pca.fits'.format(epoch, instru,
                before_or_after,sorting,channel)
        
#print('It corresponds do the path:\n{}'.format(path_im_reduced))
print('Files found:')
flist = np.sort(glob(path_im_reduced))
for f in flist: print('-',f)

CUBE_REDUCED_GOOD_PA = np.array( [ fits.getdata(flist[i]) for i in range(len(flist)) ] )

# PSF
path_im_prereduced = infolder + folder_prereduced + instru + '/*{}*/'.format(epoch) + '*median_unsat.fits'
print('\nLoad PSF file')
print('Files found:')
flist = glob(path_im_prereduced)
for f in flist: print('-',f)
im_prereduced = fits.getdata(flist[0])
print('-> Shape of the PSF', np.shape(im_prereduced))

# Wavelength grid
print('\nLoad wavelength grid')
path_im_prereduced = infolder + folder_prereduced + instru + '/*{}*/'.format(epoch) + '*lam.fits'
        
print('Files found:')
flist = glob(path_im_prereduced)
for f in flist: print('-',f)
wgrid = fits.getdata(flist[0])


## Check PSF

# Its radial profile #
plot_psf_radial_profile =  params_plot['plot_psf_radial_profile']

if plot_psf_radial_profile:
    print('\n== Plot the radial profile of the PSF ==')
    im = np.copy(im_prereduced)[0][0]
    profile_mean, separations = rad_profile(im, center=np.shape(im)[0]//2, mode='mean', pixscale=platescale*1e-3, skip_pix=1)

    # Figure
    fig, ax = plt.subplots(1,1,figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.2,right=0.95, top=0.95, bottom=0.17)

    # PSF before
    im = np.copy(im_prereduced)[0][0]
    profile_mean, separations = rad_profile(im, center=np.shape(im)[0]//2, mode='mean', pixscale=platescale*1e-3, skip_pix=1)
    ax.plot(separations/platescale*1e3, profile_mean, color='crimson',marker='D', ms=5, label='PSF before')
    # PSF after
    im = np.copy(im_prereduced)[0][1]
    profile_mean, separations = rad_profile(im, center=np.shape(im)[0]//2, mode='mean', pixscale=platescale*1e-3, skip_pix=1)
    ax.plot(separations/platescale*1e3, profile_mean, color='mediumseagreen',marker='o', ms=5, label='PSF after')

    ax.set_xlabel('Radius (pixel)')
    ax.set_ylabel('Flux (ADU)')

    ax.legend(frameon=False, loc='center right')

    text = '{} {} {}'.format(epoch, instru, filt.replace('_',''))
    ax.text(0.17,0.85,text,transform=ax.transAxes)

    namesave = 'PSF_radial_profile_flux_'+epoch+'_'+instru.replace(' ','_')
    mise_en_page(ax, x_step=1, x_maj_step=5, x_min_step=1)
    plt.savefig(saving_dir+namesave+'.pdf')
    if plot_show: plt.show()
    plt.clf()
        
# Select only one PSF #
if epoch == '2019-07-09': im_prereduced=im_prereduced[:,1] # The PSF after is the good one
elif epoch == '2016-04-05':
    im_prereduced = np.nanmean(im_prereduced,axis=0) # The PSF after is the good one
    print('WARNING, SHAPE CORRECT?', np.shape(im_prereduced))
    

# Compare star flux to the background #
print('\n== Compare star flux to the background ==')
# Define region parameters
r_star = params_config['star_r']
r_bkg_in, r_bkg_out = params_config['bkg_r_in'], params_config['bkg_r_out']
im_crop_bkg = 0

# Define Figure parameters
color_star, color_bkg = 'gold', [0,0,0.3]; color_bkg_lighter='royalblue'
ms_star, ms_bkg = 7,5
color_scalebar='black'
color_text='black'

# Plot Figure Flux wrt to the spectral channel
nlbd = np.shape(im_prereduced)[0]

fig, ax = plt.subplots(1,1,figsize=figsize_rectangle)
fig.subplots_adjust(left=0.2,bottom=0.17, top=0.95, right=0.97)
norm = 1e3

# Several PSFs
if len(np.shape(im_prereduced)) == 4:
    for i in range(np.shape(im_prereduced)[1]):
        #if i == 1 : continue
        im_i = np.copy(im_prereduced[:,i])#/norm
        if i == 1:
            flux_star, Y_bkg = shortcut_plot_flux_star_and_background(ax, im_i, r_star=r_star, r_bkg_in=r_bkg_in, r_bkg_out=r_bkg_out, 
                    ms_star=ms_star, ms_bkg=ms_bkg, color_star='w', color_bkg='w', return_flux=1,
                    do_norm=1, give_X=1, X=wgrid, display=0)
        else: 
            flux_star, Y_bkg = shortcut_plot_flux_star_and_background(ax, im_i, r_star=r_star, r_bkg_in=r_bkg_in, r_bkg_out=r_bkg_out, 
                    ms_star=ms_star, ms_bkg=ms_bkg, color_star=color_star, color_bkg=color_bkg,  return_flux=1,
                    do_norm=1, give_X=1, X=wgrid, display=0)
        
# One PSF
if len(np.shape(im_prereduced)) == 3:
    flux_star, Y_bkg = shortcut_plot_flux_star_and_background(ax, im_prereduced, r_star=r_star, r_bkg_in=r_bkg_in, r_bkg_out=r_bkg_out,
            ms_star=ms_star, ms_bkg=ms_bkg, color_star=color_star, color_bkg=color_bkg, do_norm=1,  return_flux=1,
            give_X=1, X=wgrid, display=0)
    
# Add labels
ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel(r'Flux (ADU)')

# Change size
if epoch == '2019-07-09':
    loc='upper right'
    if r_star < 6 : ax.set_ylim([-500, 2500]); y_maj_step=500; y_min_step=100
    else: ax.set_ylim([-20, 150]); y_maj_step=50; y_min_step=10
if epoch == '2016-04-05':
    if r_star < 6 : ax.set_ylim([0, 2500])
    else: ax.set_ylim([0, 400]); y_maj_step=100; y_min_step=20
    loc='upper right'
if epoch == '2016-06-03':
    if r_star < 6 : ax.set_ylim([0, 2000]); loc='lower right'
    else: ax.set_ylim([0, 150]); loc='upper right'; y_maj_step=50; y_min_step=10
    
ax.set_xlim([wgrid[0]-wgrid[0]*0.02,wgrid[-1]+wgrid[0]*0.02])

# Add text
text = '{} {} {}'.format(epoch, instru, filt.replace('_',''))
ax.text(0.06,0.87,text,transform=ax.transAxes)

# Add legend
ax.plot([-10],[0], marker='*', ms=7, label= 'star', lw=0, color='gold')
ax.plot([-10],[0], marker='o', ms=5, label= 'background', lw=0, color=[0,0,0.4])
ax.legend(loc=loc, numpoints=1, frameon=False)
          
# Save
if r_star < 6 :
    mise_en_page(ax, x_step=1, x_maj_step=5, x_min_step=1, y_step=1, y_maj_step=500, y_min_step=100)
mise_en_page(ax, x_step=1, x_maj_step=0.1, x_min_step=0.02, y_step=1, y_maj_step=y_maj_step, y_min_step=y_min_step) 
namesave = 'check_star_and_bkgd_flux_'+text.replace(' ','_')
plt.savefig(saving_dir+namesave+'.pdf')
if plot_show: plt.show()
plt.clf()
          
          
## BLOCK 4: DERIVE SPECTRUM ##
print('\n== Derive the spectrum of the object ==')
## Derive spectroscopic measurements
plot_spectrum =  params_plot['plot_spectrum']

# SCIENCE GOOD PA
cube = np.copy(CUBE_REDUCED_GOOD_PA)[0]
cube_cropped = cube[:,im_crop_pca:-im_crop_pca,im_crop_pca:-im_crop_pca]

# SCIENCE OPP PA -> derive errorbars
cube = np.copy(CUBE_REDUCED_OPP_PA)[0]
cube_cropped_err = cube[:,im_crop_pca:-im_crop_pca,im_crop_pca:-im_crop_pca]

# Consider the region given by "mask_type" parameter
mask_type = params_config['mask_type'] 

# Homogeneize image shapes
if mask_type == 'full_inner_disk':
    im_mask_cropped = np.copy(im_mask_ifs[im_crop_mask:-im_crop_mask,im_crop_mask:-im_crop_mask])

elif mask_type == 'circular':
    # and define a circular mask center at one given location (xloc,yloc)
    xloc, yloc, rcut =  params_config['mask_xm']-im_crop_pca,  params_config['mask_ym']-im_crop_pca, params_config['mask_rcut'] 
    im_mask_cropped = compute_im_rad_grid(cube_cropped[0], center='custom', xm=xloc, ym=yloc)
    im_mask_cropped = np.where(im_mask_cropped<rcut, 1, 0) 

nb_pix_used = len(im_mask_cropped[im_mask_cropped==1])
print('Number of pixels considered for the mask:', nb_pix_used)
cube = cube_cropped * im_mask_cropped
cube_err = cube_cropped_err * im_mask_cropped

# Derive flux values within the mask
cond = cube > 0
cube[~cond] = np.nan
flux = np.nansum(cube, axis=(1,2)) / nb_pix_used 

# Derive noise within the mask
cond = cube_err > 0
cube_err[~cond] = np.nan
std = np.nanstd(cube_err, axis=(1,2))

platescale2arcsec = (platescale*1e-3)**2

print(np.shape(cube_err))

if writeto:
    fits.writeto('fits/cube_{}_{}_all_channels_mask_{}_good_pa.fits'.format(epoch, instru, mask_type), cube, overwrite=True)
    fits.writeto('fits/cube_{}_{}_all_channels_mask_{}_opp_pa.fits'.format(epoch, instru, mask_type), cube_err, overwrite=True)

if plot_spectrum:
    # Figure spectrum (flux ADU / pixel)
    namesave='spectrum_fluxADU_{}_{}_all_channels_mask_{}.pdf'.format(epoch, instru, mask_type)

    X = wgrid
    norm = 1 * platescale2arcsec

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.2,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Flux (ADU/arcsec$^2$)')

    text = '{}'.format(epoch, filt)
    ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.errorbar(X, flux/norm, std/norm,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.05, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)

    ylim = ax.get_ylim()
    ylim = [0,ylim[1]]
    ax.set_ylim(ylim)
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()


    # Figure spectrum (contrast)
    namesave='spectrum_contrast_{}_{}_all_channels_mask_{}.pdf'.format(epoch, instru, mask_type)

    X = wgrid
    norm = flux_star 

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.12,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Contrast')

    text = '{}'.format(epoch, filt)
    ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.errorbar(X, flux/norm, std/norm,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.05, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)
    ylim = ax.get_ylim()
    ylim = [0,ylim[1]]
    ax.set_ylim(ylim)
    
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()

    # Figure spectrum (flux (W/m^2))
    namesave='spectrum_fluxWm2_{}_{}_all_channels_mask_{}.pdf'.format(epoch, instru, mask_type)
    fn_star_spectrum_syn = params_path['fn_star_spectrum_syn']

    X = wgrid
    Y_star_syn = synthetic_spectrum_star2wgrid_obs(fn_star_spectrum_syn, wgrid, saving_dir=saving_dir, show=0, plot=1)
    norm = Y_star_syn / flux_star

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.15,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)');  ax.set_ylabel('Flux (W/m$^2$)')

    text = '{}'.format(epoch, filt)
    ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.errorbar(X, flux/norm, std/norm,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.05, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)
    ylim = ax.get_ylim()
    ylim = [0,ylim[1]]
    ax.set_ylim(ylim)
    
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()


