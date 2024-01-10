# Previous Version 2023-05-23 (beg. day)
# Old way to extract flux within an aperture
# Next version: I will use the astropy function, more accurate to extract the flux within a circle and not a square (in particular for small apertures)
# Now, new version

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
from functions_binning import *

# Update Figure outline
#plt.style.use('classic')  
#mpl.rc('image', cmap='viridis', origin='lower')

#rcParams.update({'font.size': 12,  'axes.labelsize' : 12, 'legend.fontsize' : 12,
#                 "font.family": "serif", 'text.usetex' : True, "font.serif": [], "font.sans-serif": [], 'mathtext.fontset':"stix",
#'legend.handlelength': 1.5, 'legend.borderaxespad' : 0.8, 'legend.columnspacing' : 1, 'legend.handletextpad' : 0.5})

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

## Read config file
parser = argparse.ArgumentParser(description='derive spectrum')
parser.add_argument("config", type=str, help='configuration file')
args = parser.parse_args()
print("\nRead the yaml file: {}\n".format(args.config))
with open(args.config,"r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params_config = params['config'] 
 
## Initialize saving directory
L = time.localtime()
date = "{}-{}-{}".format(L[0],L[1],L[2],L[3],L[4],L[5])

saving_dir = 'figs/fig_v{}/'.format(date)
saving_dir_root = saving_dir
os.makedirs(saving_dir,exist_ok=True)


# Subfolder in which to save the files
mask_type, mask_name = params_config['mask_type'], params_config['mask_name']
loc_x, loc_y = params_config['mask_xm'],  params_config['mask_ym']

if mask_type == 'fits_file': folder = mask_name+'/'
else: folder = 'loc_x={}_y={}pix/'.format(loc_x, loc_y)

saving_dir += folder
os.makedirs(saving_dir,exist_ok=True)


## Functions
def aperture_photometry_cube(cube, xloc, yloc, rcut, return_std=0, norm=1):
    n = np.shape(cube)[0]
    FLUX, STD = [], []
    # Define aperture
    positions = [(xloc, yloc)]
    aperture = CircularAperture(positions, r=rcut)
    
    # Derive flux
    for i, im in enumerate(cube):
        phot_table = aperture_photometry(im, aperture)
        flux = phot_table['aperture_sum'].value[0]
    
        # Divide by the aperture size
        nb_pix_used = np.pi * (rcut*norm)**2
        flux /= nb_pix_used
        FLUX.append(flux)
        
        # Standard deviation
        aperstats = ApertureStats(im, aperture)
        std = aperstats.std[0]/(norm**2) # not sure I should divide by norm, norm represent the binning factor. But otherwise my errorbars explode
        # If the noise was proportional to the plate scale, I could divide by it
        # But they are several sources of noise
        # photon noise, prop to the platescale
        # 
        STD.append(std)

    FLUX, STD = np.array(FLUX), np.array(STD)
    if return_std: return FLUX, STD
    return FLUX



def plot_mosaic_wgrid(cube, wgrid, epoch, saving_dir='', namesave='', im_crop=0, show=0,
                      add_obj = 0, add_cbar=1, pixarc = 7.46, scalebar_sep = 0.1):

    print('Display a mosaic of the IFS images, as a function of the wavelength.')

    n = len(cube)

    vmax = np.nanmax(cube)*0.3
   
    if vmax > 0.95e-06 : vmax = 9e-7
    elif vmax > 5e-07 and vmax < 5.5e-07: vmax = 6e-7
    elif vmax > 1.5e-07 and vmax < 5e-07: vmax = 3e-7
    
    vmax = float('%.e' % vmax)
    vmin = -vmax

    # Figsize

    if spectral_binning_factor == 1: nli = 5
    elif spectral_binning_factor == 2 : nli = 4
    elif spectral_binning_factor >= 4 : nli = 2
    else : nli = 3

    nco, num = n//nli, -1
    if n%nli != 0: nco

    fig, axx = plt.subplots(nli,nco,figsize=(2*nco+1,2*nli))
    fig.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.87,hspace=0.015,wspace=0.015)
    
    for li in range(nli):
        for co in range(nco):
            ax = axx[li,co] ; ax.axis('off') ; num +=  1
        
            if num >= n : continue
            if im_crop != 0: im = cube[num, im_crop:-im_crop,  im_crop:-im_crop]
            else: im = cube[num]
    
            im_plot = ax.imshow(im,origin='lower', cmap = 'magma', interpolation='nearest', 
                            vmin=vmin,vmax=vmax)
            h,w = im.shape 
   
            # add text label
            text = '{}  {:.3f}$\,\mu$m'.format(epoch, wgrid[num])
            ax.text(0.02,0.9,text,color='white',transform=ax.transAxes)
        
            # add scale
            #scalebar_au = '{} au'.format(round_sig(sep2au(scalebar_sep,dist)))
            scalebar_as = '{}"'.format(scalebar_sep)
            #ax.text(w*0.8,h*0.13,'{} au'.format('%.1f'%sep2au(scalebar_sep,distobj)),color='white')
            scalebar = AnchoredSizeBar(ax.transData, scalebar_sep*1e3/pixarc, scalebar_as, 4, 
            label_top=True, pad=0.2, sep=3, borderpad=0.3, frameon=False, size_vertical=1, color='white')
            ax.add_artist(scalebar)

            if add_obj :
                pass

            if add_cbar :
                orientation = 'vertical'
                
                cb_xdeb, cb_ydeb, cb_xwidth, cb_ywidth = 0.89, 0.05, 0.015, 0.9  
                vmax_cbar = vmax
                cb_ax = fig.add_axes([cb_xdeb, cb_ydeb, cb_xwidth, cb_ywidth])
                cbar_bound = np.linspace(vmin,vmax_cbar,7)
    
                fff = mticker.ScalarFormatter(useMathText=True)
                fff.set_powerlimits((-10,10))
                cbar_labels = []
                for val in cbar_bound:
                    if val == 1e-6: cbar_labels.append(r"$1\times10^{-6}$")
                    elif val == 1e-7: cbar_labels.append(r"$1\times10^{-7}$")
                    elif val == 1e-8: cbar_labels.append(r"$1\times10^{-8}$")
                    elif val !=0: cbar_labels.append("${}$".format(fff.format_data(val))) #int(val*10/vmax)*vmax/10)))
                    else: cbar_labels.append("0")

                cbar = fig.colorbar(im_plot, cax=cb_ax,orientation=orientation, boundaries=cbar_bound, 
                        drawedges=True, pad=0.8, shrink=0.05,fraction=0.01,aspect=50)#,width_fraction=0.001)
                cbar.set_ticklabels(cbar_labels)

            
    if namesave != '':
        plt.savefig(saving_dir+namesave+'.pdf',dpi=300)
    if show: plt.show()
    return


def synthetic_spectrum_star2wgrid_obs(fn_syn, w_obs, plot=0, saving_dir='', show=1, change_units=0, apply_LF=0):
    '''
    Return the synthetic spectrum (nominal file "fn_syn") on the wavelength grid
    of the observation ("fn_wgrid_obs").
    '''
    ## Load obs
    # w_obs = fits.getdata(fn_wgrid_ifs)
    
    # Load synthethic file
    
    #file = np.loadtxt(fn_syn)
    #w, f = file[:,0], file[:,1]

    file = fits.getdata(fn_syn)
    w, f = file[0], file[1]

    if change_units:
        # Add units
        w_units = w * u.Angstrom
        f_units = f * units.erg / u.s / u.cm**2 / u.Angstrom

        # Remove the wavelength in the flux
        f_units *= w_units

        # Convert (!) problem in the conversion I think ?!
        f_units = f_units.to(u.W / u.m **2)
        w_units = w_units.to(u.micron)

        w, f = w_units.value, f_units.value

    # Restrict the region of interest
    cond_w = np.logical_and(w > 0.9, w < 1.6)

    X, Y = w[cond_w], f[cond_w]

    if apply_LF:
        # Remove high frequencies
        Y_BF = savgol_filter(Y,window_length=2000, polyorder=3, mode='nearest')

        # Do it on the wavelength grid of the observation
        Y_OBS = griddata(X, Y_BF, w_obs)
        
    else: Y_OBS = griddata(X, Y, w_obs)
    X_OBS = w_obs

    if plot:
        # Figure
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        ax.semilogy(X, Y, label='synthetic spectrum', color = 'dodgerblue')
        if apply_LF: ax.semilogy(X, Y_BF, label='low-filtered synthethic spectrum', color='navy', lw=2)
        ax.semilogy(X_OBS, Y_OBS, label='expected for obs.', color='gold', lw=0, marker='*', ms=8)
        ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Flux (W/m$^2$)')
        ax.legend(loc=0, numpoints=1)
        mise_en_page(ax)
        plt.tight_layout()
        plt.savefig(saving_dir+'synthetic_spectrum.pdf')
        if show: plt.show()
        plt.clf()

    if plot:
        # Figure bis
        fig, ax = plt.subplots(1,1, figsize=(4,3))
        fig.subplots_adjust(left=0.2,right=0.96,bottom=0.2,top=0.93)
        ax.plot(X_OBS, Y_OBS, label='stellar synthetic spectrum', color='gold', lw=1, marker='*', ms=8)
        ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Flux (W/m$^2$)')
        ax.legend(loc=0, numpoints=1, frameon=False)
        mise_en_page(ax)

        mise_en_page(ax,x_step=1, x_maj_step=0.1, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)
        xlim = [0.95,1.35]

        ax.set_xlim(xlim)
    
        #plt.tight_layout()
        plt.savefig(saving_dir+'synthetic_spectrum_star_IFS.pdf')
        if show: plt.show()
        plt.clf()
    
    return Y_OBS

##############################################
## Block 1: INITIALIZATION FOR THIS SYSTEM ##
##############################################

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
im_crop_pca = 8*5 #9 #im_crop_PCA
im_crop_mask = 8*5


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

#######################################
## BLOCK 3: LOAD ONE SPECIFIC EPOCH ##
#######################################

## Load the files to derive the spectrum of the object
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
wgrid_orig = np.copy(wgrid)


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

# Apply spectral binning ?
spectral_binning_factor = params_config['spectral_binning_factor']
if spectral_binning_factor != 1 : 
    im_prereduced, wgrid = binning_spectral(im_prereduced, wgrid_orig, spectral_binning_factor=spectral_binning_factor)
    

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
    
ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]*spectral_binning_factor])
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
#if plot_show: plt.show()
plt.clf()
          


###############################
## BLOCK 4: SPATIAL BINNING ##
###############################
# SCIENCE GOOD PA
cube = np.copy(CUBE_REDUCED_GOOD_PA)[0]
cube = cube[:,im_crop_pca:-im_crop_pca,im_crop_pca:-im_crop_pca]

# SCIENCE OPP PA -> derive errorbars
cube_err = np.copy(CUBE_REDUCED_OPP_PA)[0]
cube_err = cube_err[:,im_crop_pca:-im_crop_pca,im_crop_pca:-im_crop_pca]

# invert XY axis
cube = cube[:,::-1,::-1]
cube_err = cube_err[:,::-1,::-1]

print('\n== Bin spatially ==')
idx_w_ref = params_config['idx_w_ref']
spatial_binning_factor = params_config['spatial_binning_factor']

# Mask
# Consider the region given by "mask_type" parameter
xloc, yloc  =  (params_config['mask_xm']-im_crop_pca)/spatial_binning_factor,  (params_config['mask_ym']-im_crop_pca)/spatial_binning_factor
rcut = params_config['mask_rcut']/spatial_binning_factor
xloc_orig, yloc_orig  =  (params_config['mask_xm']-im_crop_pca),  (params_config['mask_ym']-im_crop_pca)
rcut_orig = params_config['mask_rcut']


print('The binning factor is {}x{} pixels^2'.format(spatial_binning_factor, spatial_binning_factor))
if spatial_binning_factor != 1 :
    print('Apply binning')
    namesave = 'binning_spatial_{}x{}pix'.format(spatial_binning_factor,spatial_binning_factor)
    func = np.sum
    
    # Resample
    cube_binned = block_reduce_cube(cube, spatial_binning_factor, func=func)
    cube_binned_err = block_reduce_cube(cube_err, spatial_binning_factor, func=func)
    
    # Figure
    TEXTS = ['original', 'spatial binning {}x{}'.format(spatial_binning_factor, spatial_binning_factor)]
    IM = np.array([cube[idx_w_ref], cube_binned[idx_w_ref] ])
    vmax = np.nanmax(IM[0])
    for im in IM: print('Shape of before/after binning:', np.shape(im))


    if mask_type == 'circular' and 0: # 2 Figs not only 1
        # Figure
        #TEXTS = ['original', 'spatial binning {}x{}'.format(spatial_binning_factor, spatial_binning_factor)]
        #print(int(1+idx_w_ref/spectral_binning_factor))
        #print(int(idx_w_ref/spectral_binning_factor))
        #print(1+idx_w_ref/spectral_binning_factor)
        IM = np.array(cube[int(idx_w_ref/spectral_binning_factor)])
        vmax = np.nanmax(IM)
        print('Shape of the image', np.shape(IM))

        text = '{} IFS {:.3f}$\,\mu$m'.format(epoch, wgrid[int(idx_w_ref/spectral_binning_factor)])
        print(text)
    
        fig, ax = plot_fig(IM, add_colorbar=0, figsize=(2.5,2.5), add_text=1, text=text,
                           do_tight_layout=1, flux_use_binning_factor=1, vmax=vmax, vmin=-vmax,  scalebar_loc=4, 
                           namesave=namesave, saving_dir=saving_dir, 
                           scalebar_leg='0.1"', show=plot_show, return_fig_ax=1)
    

        # Mask
        im_mask_cropped = np.copy(im_mask_ifs[im_crop_mask:-im_crop_mask,im_crop_mask:-im_crop_mask])
        nb_pix_used = len(im_mask_cropped[im_mask_cropped==1])
    
        # Contour mask in image orig
        n_final = np.shape(cube)[-1]
        imc = im_mask_cropped
        print('Size image mask', np.shape(imc))
        lvl_contours=0
        XX, YY = np.arange(n_final), np.arange(n_final)
        contours = ax.contour(XX, YY, imc, lvl_contours, cmap=cmap, linewidths=0.5, linestyles='-')

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
        plt.savefig(saving_dir+namesave+'.pdf',dpi=300)

        plt.clf()

    elif mask_type == 'circular':
        fig, axs = plot_figs_1row(IM, add_colorbar=0, add_text=1, text=TEXTS, figsize=(6,3),
                                  do_tight_layout=1, flux_use_binning_factor=1, vmax=vmax, vmin=-vmax,  scalebar_loc=4, 
                                  namesave=namesave, saving_dir=saving_dir, 
                                  scalebar_leg='0.1"', platescale_use_binning_factor=1, binning_factor=[1, spatial_binning_factor], show=plot_show,
                              return_fig_ax=1)
        # Circle in image orig
        circ = Circle((xloc_orig, yloc_orig), rcut_orig, edgecolor='white', fc='None', lw=0.5)
        axs[0].add_patch(circ)

        rsize= 6
        #position = (xloc_orig, yloc_orig)
        #size = (rcut_orig*rfact, rcut_orig*rfact)     # pixels
        #size_mid = (size[0]/2-1+xloc%1, size[1]/2+yloc%1)

        xzoom_deb,  xzoom_fin = 1+int(xloc_orig-rsize), 1+int(xloc_orig+rsize)
        yzoom_deb,  yzoom_fin = 1+int(yloc_orig-rsize), 1+int(yloc_orig+rsize)
    
        size = (xzoom_fin-xzoom_deb, yzoom_fin-yzoom_deb) # pixels
        size_mid = (rsize+xloc_orig%1-1, rsize+yloc_orig%1-1)

        cutout = IM[0][ yzoom_deb:yzoom_fin, xzoom_deb:xzoom_fin]
        #cutout = Cutout2D(IM[0], position, size).data  # do the zoom in
        xdeb, ydeb, xlong, ylong = -0.06, 0.035, 0.34, 0.34
        axes2 = fig.add_axes([xdeb, ydeb, xlong, ylong]) # renvoie un objet Axes
        axes2.imshow(cutout, interpolation='nearest', origin='lower', vmax=vmax,vmin=-vmax, cmap='magma')
        axes2.axis('off')
        circ = Circle(size_mid, rcut_orig, edgecolor='white', fc='None', lw=0.5)
        axes2.add_patch(circ)
        rect = Rectangle((-0.5,-0.5), size[0], size[1], edgecolor='black', fc='None', lw=2)
        axes2.add_patch(rect)

        # add scalebar in the insert
        scalebar_pix,  scalebar_leg = 10, '0.01"'
        size = scalebar_pix/(platescale)
        scalebar = AnchoredSizeBar(axes2.transData, size, scalebar_leg, 1, label_top=True, pad=0.1,
                                   sep=1, borderpad=0.1, frameon=False, size_vertical=1, color='white')#,fontproperties=fp)
        axes2.add_artist(scalebar)
    
        # Circle in the image binned
        circ = Circle((xloc, yloc), rcut, edgecolor='white', fc='None', lw=0.5)
        axs[1].add_patch(circ)

        rfact= 6/spatial_binning_factor
        rsize = 6/spatial_binning_factor
        #le flux est calculé dans l'ouverture circulaire blanche.
        #position = (xloc, yloc)
        print(xloc, yloc)
        xzoom_deb,  xzoom_fin = 1+int(xloc-rsize), 1+int(xloc+rsize)
        yzoom_deb,  yzoom_fin = 1+int(yloc-rsize), 1+int(yloc+rsize)
    
        size = (xzoom_fin-xzoom_deb, yzoom_fin-yzoom_deb) # pixels
        size_mid = (rsize+xloc%1-1, rsize+yloc%1-1)
        print(size, size_mid)
        print(2*rcut*rfact)
    
        cutout = IM[1][ yzoom_deb:yzoom_fin, xzoom_deb:xzoom_fin]
        #Cutout2D(IM[1], position, size).data # do the zoom in
        xdeb = 0.43
        axes2 = fig.add_axes([xdeb, ydeb, xlong, ylong]) # renvoie un objet Axes
        axes2.imshow(cutout, interpolation='nearest', origin='lower', vmax=vmax*spatial_binning_factor**2,vmin=-vmax*spatial_binning_factor**2, cmap='magma')
        axes2.axis('off')
   
        circ = Circle(size_mid, rcut, edgecolor='white', fc='None', lw=0.5)
        axes2.add_patch(circ)
        rect = Rectangle((-0.5,-0.5), size[0], size[1], edgecolor='black', fc='None', lw=2)
        axes2.add_patch(rect)

        # add scalebar in the insert
        scalebar_pix,  scalebar_leg = 10, '0.01"'
        size = scalebar_pix/(platescale*spatial_binning_factor)
        scalebar = AnchoredSizeBar(axes2.transData, size, scalebar_leg, 1, label_top=True, pad=0.1,
                                   sep=1, borderpad=0.1, frameon=False, size_vertical=1/spatial_binning_factor, color='white')#,fontproperties=fp)
        axes2.add_artist(scalebar)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)

        plt.savefig(saving_dir+namesave+'.pdf', dpi=300)
        #plt.show()
        plt.clf()
    
        cube, cube_err = cube_binned, cube_binned_err

    
#if writeto:
#    fits.writeto('fits/binning_spatial={}x{}pix.fits'.format(spatial_binning_factor,spatial_binning_factor), cube, overwrite=True)
#    fits.writeto('fits/binning_spatial={}x{}pix_cube_opp_pa.fits'.format(spatial_binning_factor,spatial_binning_factor), cube_err, overwrite=True)


    
################################
## BLOCK 5: SPECTRAL BINNING ##
################################
if spectral_binning_factor != 1:
    # Cube good pa
    cube, wgrid = binning_spectral(cube, wgrid_orig, spectral_binning_factor=spectral_binning_factor)
# Figure Mosaic
namesave = 'mosaic_{}_spatial_binning={}x{}pix_spectral_binning={}'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)
plot_mosaic_wgrid(cube, wgrid, epoch, im_crop=0, pixarc=platescale,
        namesave=namesave, saving_dir=saving_dir, show=plot_show)

if spectral_binning_factor != 1:
    # Cube opp pa
    cube_err, wgrid = binning_spectral(cube_err, wgrid_orig, spectral_binning_factor=spectral_binning_factor)
# Figure Mosaic
namesave = 'mosaic_{}_spatial_binning={}x{}pix_spectral_binning={}_cube_opp_pa'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)
plot_mosaic_wgrid(cube_err, wgrid, epoch, im_crop=0, pixarc=platescale,
        namesave=namesave, saving_dir=saving_dir, show=plot_show)


if writeto:
    fits.writeto('fits/binning_spatial={}x{}pix_binning_spectral={}.fits'.format(spatial_binning_factor,spatial_binning_factor, spectral_binning_factor), cube, overwrite=True)
    fits.writeto('fits/binning_spatial={}x{}pix_binning_spectral={}_cube_opp_pa.fits'.format(spatial_binning_factor,spatial_binning_factor, spectral_binning_factor), cube_err, overwrite=True)

    

if spatial_binning_factor == 1 and mask_type == 'circular':
    namesave = 'binning_spatial_{}x{}_binning_spectral={}'.format(spatial_binning_factor,spatial_binning_factor,  spectral_binning_factor)
    func = np.sum
    
    # Figure
    #TEXTS = ['original', 'spatial binning {}x{}'.format(spatial_binning_factor, spatial_binning_factor)]
    #print(int(1+idx_w_ref/spectral_binning_factor))
    #print(int(idx_w_ref/spectral_binning_factor))
    #print(1+idx_w_ref/spectral_binning_factor)
    IM = np.array(cube[int(idx_w_ref/spectral_binning_factor)])
    IM_err = np.array(cube_err[int(idx_w_ref/spectral_binning_factor)])
    vmax = np.nanmax(IM)
    print('Shape of the image', np.shape(IM))

    text = '{} IFS {:.3f}$\,\mu$m'.format(epoch, wgrid[int(idx_w_ref/spectral_binning_factor)])
    print(text)

    fig, ax = plot_fig(IM, add_colorbar=0, figsize=(2.5,2.5), add_text=1, text=text,
        do_tight_layout=1, flux_use_binning_factor=1, vmax=vmax, vmin=-vmax,  scalebar_loc=4, 
        namesave=namesave, saving_dir=saving_dir, 
        scalebar_leg='0.1"', show=plot_show, return_fig_ax=1)
    
    # Circle in image orig
    circ = Circle((xloc_orig, yloc_orig), rcut_orig, edgecolor='white', fc='None', lw=0.5)
    ax.add_patch(circ)

    rsize= 6
    xzoom_deb,  xzoom_fin = 1+int(xloc_orig-rsize), 1+int(xloc_orig+rsize)
    yzoom_deb,  yzoom_fin = 1+int(yloc_orig-rsize), 1+int(yloc_orig+rsize)
    
    size = (xzoom_fin-xzoom_deb, yzoom_fin-yzoom_deb) # pixels
    size_mid = (rsize+xloc_orig%1-1, rsize+yloc_orig%1-1)

    cutout = IM[ yzoom_deb:yzoom_fin, xzoom_deb:xzoom_fin]
    xdeb, ydeb, xlong, ylong = 0.05, 0.035, 0.3, 0.3
    axes2 = fig.add_axes([xdeb, ydeb, xlong, ylong]) # renvoie un objet Axes
    axes2.imshow(cutout, interpolation='nearest', origin='lower', vmax=vmax,vmin=-vmax, cmap='magma')
    axes2.axis('off')
    circ = Circle(size_mid, rcut_orig, edgecolor='white', fc='None', lw=0.5)
    axes2.add_patch(circ)
    rect = Rectangle((-0.5,-0.5), size[0], size[1], edgecolor='black', fc='None', lw=2)
    axes2.add_patch(rect)

    # add scalebar in the insert
    scalebar_pix,  scalebar_leg = 10, '0.01"'
    size = scalebar_pix/(platescale)
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(size=12)
    
    scalebar = AnchoredSizeBar(axes2.transData, size, scalebar_leg, 1, label_top=True, pad=0.1,
                                   sep=1, borderpad=0.1, frameon=False, size_vertical=1, color='white', fontproperties=fp)
    axes2.add_artist(scalebar)
  

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
    plt.savefig(saving_dir+namesave+'.pdf', dpi=300)

    plt.clf()

    


elif spatial_binning_factor == 1 and mask_type == 'fits_file':
    namesave = 'binning_spatial_{}x{}_binning_spectral={}'.format(spatial_binning_factor,spatial_binning_factor,  spectral_binning_factor)
    func = np.sum
    
    # Figure
    #TEXTS = ['original', 'spatial binning {}x{}'.format(spatial_binning_factor, spatial_binning_factor)]
    #print(int(1+idx_w_ref/spectral_binning_factor))
    #print(int(idx_w_ref/spectral_binning_factor))
    #print(1+idx_w_ref/spectral_binning_factor)
    IM = np.array(cube[int(idx_w_ref/spectral_binning_factor)])
    IM_err = np.array(cube_err[int(idx_w_ref/spectral_binning_factor)])
    vmax = np.nanmax(IM)
    print('Shape of the image', np.shape(IM))

    text = '{} IFS {:.3f}$\,\mu$m'.format(epoch, wgrid[int(idx_w_ref/spectral_binning_factor)])
    print(text)

    fig, ax = plot_fig(IM, add_colorbar=0, figsize=(2.5,2.5), add_text=1, text=text,
        do_tight_layout=1, flux_use_binning_factor=1, vmax=vmax, vmin=-vmax,  scalebar_loc=4, 
        namesave=namesave, saving_dir=saving_dir, 
        scalebar_leg='0.1"', show=plot_show, return_fig_ax=1)
    

    # Mask
    im_mask_cropped = np.copy(im_mask_ifs[im_crop_mask:-im_crop_mask,im_crop_mask:-im_crop_mask])
    nb_pix_used = len(im_mask_cropped[im_mask_cropped==1])
    
    # Contour mask in image orig
    n_final = np.shape(cube)[-1]
    imc = im_mask_cropped
    print('Size image mask', np.shape(imc))
    lvl_contours=0
    XX, YY = np.arange(n_final), np.arange(n_final)
    contours = ax.contour(XX, YY, imc, lvl_contours, cmap=cmap, linewidths=0.5, linestyles='-')
    #ax.clabel(contours, inline=True) 

    if 0: # Insert - not useful if the region is large enough
        rsize= 20
        xzoom_deb,  xzoom_fin = 1+int(xloc_orig-rsize), 1+int(xloc_orig+rsize) #!
        yzoom_deb,  yzoom_fin = 1+int(yloc_orig-rsize), 1+int(yloc_orig+rsize)

        size = (xzoom_fin-xzoom_deb, yzoom_fin-yzoom_deb) # pixels
        size_mid = (rsize+xloc_orig%1-1, rsize+yloc_orig%1-1)

        cutout = IM[ yzoom_deb:yzoom_fin, xzoom_deb:xzoom_fin]
        cutout_mask = im_mask_cropped[ yzoom_deb:yzoom_fin, xzoom_deb:xzoom_fin]
        
        xdeb, ydeb, xlong, ylong = 0.05, 0.035, 0.3, 0.3
        axes2 = fig.add_axes([xdeb, ydeb, xlong, ylong]) # renvoie un objet Axes
        axes2.imshow(cutout, interpolation='nearest', origin='lower', vmax=vmax,vmin=-vmax, cmap='magma')
        axes2.axis('off')

        # contours
        n_final = np.shape(cutout)[-1]
        imc = cutout_mask
        print('Size image mask', np.shape(imc))
        lvl_contours=7
        XX, YY = np.arange(n_final), np.arange(n_final)
        contours = axes2.contour(XX, YY, imc, lvl_contours, cmap=cmap, linewidths=0.2, linestyles='-')
        
        # Rectangle framework
        rect = Rectangle((-0.5,-0.5), size[0], size[1], edgecolor='black', fc='None', lw=2)
        axes2.add_patch(rect)

        # add scalebar in the insert
        scalebar_pix,  scalebar_leg = 10, '0.01"'
        size = scalebar_pix/(platescale)
        from matplotlib.font_manager import FontProperties

        fp = FontProperties(size=12)
    
        scalebar = AnchoredSizeBar(axes2.transData, size, scalebar_leg, 1, label_top=True, pad=0.1,
                                   sep=1, borderpad=0.1, frameon=False, size_vertical=1, color='white', fontproperties=fp)
        axes2.add_artist(scalebar)
  

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
    plt.savefig(saving_dir+namesave+'.pdf',dpi=300)

    plt.clf()


if writeto:
    fits.writeto(saving_dir_root+'im_binning_spatial={}x{}pix_binning_spectral={}.fits'.format(spatial_binning_factor,spatial_binning_factor,spectral_binning_factor),
                 IM, overwrite=True)
    fits.writeto(saving_dir_root+'im_binning_spatial={}x{}pix_binning_spectral={}_cube_opp_pa.fits'.format(spatial_binning_factor,spatial_binning_factor,spectral_binning_factor),
                 IM_err, overwrite=True)

    



###############################
## BLOCK 6: DERIVE SPECTRUM ##
###############################

print('\n== Derive the spectrum of the object ==')
## Derive spectroscopic measurements
plot_spectrum =  params_plot['plot_spectrum']

# Homogeneize image shapes
if mask_type == 'fits_file':
    im_mask_cropped = np.copy(im_mask_ifs[im_crop_mask:-im_crop_mask,im_crop_mask:-im_crop_mask])

    nb_pix_used = len(im_mask_cropped[im_mask_cropped==1])
    cube = cube * im_mask_cropped # 'cube' has already been cropped
    cube_err = cube_err * im_mask_cropped

    # Derive flux values within the mask
    cond = cube > 0
    cube[~cond] = np.nan
    flux = np.nansum(cube, axis=(1,2)) / nb_pix_used 

    # Derive noise within the mask
    cond = cube_err > 0
    cube_err[~cond] = np.nan
    std = np.nanstd(cube_err, axis=(1,2))

elif mask_type == 'circular':
    # Define a circular aperture centered at one given location (xloc,yloc)
    # Use the astropy function
    nb_pix_used = np.pi * rcut**2

    print('The flux is estimated within a circular aperture (x,y) equal to ({:.1f},{:.1f}) of radius = {:.1f} pixels'.format(xloc, yloc, rcut))

    # Derive flux values within the aperture
    cond_neg = cube <= 0
    cube_pos = np.copy(cube)
    #cube_pos[cond_neg] = 0
    flux = aperture_photometry_cube(cube_pos, xloc, yloc, rcut, return_std=0, norm=spatial_binning_factor)
    dum, std = aperture_photometry_cube(cube_err, xloc, yloc, rcut+3, return_std=1, norm=spatial_binning_factor)

print('Number of pixels considered for the mask:', nb_pix_used)
print('Flux estimated:\n', flux)
print('Standard deviation estimated:\n', std)
     


platescale2arcsec = (platescale*1e-3)**2

print(np.shape(cube_err))

if writeto:
    fits.writeto('fits/cube_{}_{}_all_channels_mask_{}_good_pa.fits'.format(epoch, instru, mask_type), cube, overwrite=True)
    fits.writeto('fits/cube_{}_{}_all_channels_mask_{}_opp_pa.fits'.format(epoch, instru, mask_type), cube_err, overwrite=True)

if plot_spectrum:
    # Figure spectrum (flux ADU / pixel)
    namesave='spectrum_fluxADU_{}_{}_all_channels_mask_{}_spatial_binning={}x{}pix_spectral_binning={}.pdf'.format(epoch, instru, mask_type,
                    spatial_binning_factor, spatial_binning_factor,  spectral_binning_factor)

    X = wgrid
    norm = 1 * platescale2arcsec

    Y_ADUarcsec2, Y_err_ADUarcsec2 = np.copy(flux/norm), np.copy(std/norm)

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.2,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Flux (ADU/arcsec$^2$)')

    
    if spatial_binning_factor == 1 and spectral_binning_factor == 1:
        text = '{} \nno spatial binning\nno spectral binning'.format(epoch, spectral_binning_factor)
    elif spatial_binning_factor == 1:
        text = '{} \nno spatial binning\nspectral binning {} channels'.format(epoch, spectral_binning_factor)
    elif spectral_binning_factor == 1:
        text = '{} \nspatial binning {}x{} pixels$^2$ \nno spectral binning'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)  
    else:
        text = '{} \nspatial binning {}x{} pixels$^2$ \nspectral binning {} channels'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)
        
    #ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.text(0.15, 0.7,text, transform=ax.transAxes)
    
    ax.errorbar(X, Y_ADUarcsec2, Y_err_ADUarcsec2,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.1, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)

    ylim = ax.get_ylim()
    ylim = [0,ylim[1]]
    ylim = [0,0.003*spectral_binning_factor]

    xlim = [0.95,1.35]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()


    # Figure spectrum (contrast)
    namesave='spectrum_contrast_{}_{}_all_channels_mask_{}_spatial_binning={}x{}pix_spectral_binning={}.pdf'.format(epoch, instru, mask_type,
                    spatial_binning_factor, spatial_binning_factor,  spectral_binning_factor)

    X = wgrid
    norm = flux_star
    Y_contr, Y_err_contr =  np.copy(flux/norm), np.copy(std/norm),

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.14,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)'); ax.set_ylabel('Contrast')

    if spatial_binning_factor == 1 and spectral_binning_factor == 1:
        text = '{} \nno spatial binning\nno spectral binning'.format(epoch, spectral_binning_factor)
    elif spatial_binning_factor == 1:
        text = '{} \nno spatial binning\nspectral binning {} channels'.format(epoch, spectral_binning_factor)
    elif spectral_binning_factor == 1:
        text = '{} \nspatial binning {}x{} pixels$^2$ \nno spectral binning'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)  
    else:
        text = '{} \nspatial binning {}x{} pixels$^2$ \nspectral binning {} channels'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)
        
    #ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.text(0.15,0.7,text, transform=ax.transAxes)
    
    ax.errorbar(X, Y_contr, Y_err_contr,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.1, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)
    ylim = ax.get_ylim()
    #ylim = [0,ylim[1]]
    ylim = [0,0.3e-8]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()

    # Figure spectrum (flux (W/m^2))
    namesave='spectrum_fluxWm2_{}_{}_all_channels_mask_{}_spatial_binning={}x{}pix_spectral_binning={}.pdf'.format(epoch, instru, mask_type,
                    spatial_binning_factor, spatial_binning_factor,  spectral_binning_factor)
    fn_star_spectrum_syn = params_path['fn_star_spectrum_syn']

    X = wgrid
    Y_star_syn = synthetic_spectrum_star2wgrid_obs(fn_star_spectrum_syn, wgrid, saving_dir=saving_dir, show=0, plot=1)
    norm = flux_star / Y_star_syn

    Y_Wm2, Y_err_Wm2 = np.copy(flux/norm), np.copy(std/norm)

    print(Y_Wm2)
     
    print(Y_err_Wm2)

    fig, ax = plt.subplots(1,1, figsize=figsize_rectangle)
    fig.subplots_adjust(left=0.2,right=0.96,bottom=0.17,top=0.93)
    ax.set_xlabel('Wavelength ($\mu$m)');  ax.set_ylabel('Flux (W/m$^2$)')

    text = '{}'.format(epoch, filt)

    print(spatial_binning_factor, spectral_binning_factor )
    if spatial_binning_factor == 1 and spectral_binning_factor == 1:
        print('1 1')
        text = '{} \nno spatial binning\nno spectral binning'.format(epoch, spectral_binning_factor)
    elif spatial_binning_factor == 1:
        print('1 spa')
        text = '{} \nno spatial binning\nspectral binning {} channels'.format(epoch, spectral_binning_factor)
    elif spectral_binning_factor == 1:
        print('1 spec')
        text = '{} \nspatial binning {}x{} pixels$^2$ \nno spectral binning'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)  
    else:
        text = '{} \nspatial binning {}x{} pixels$^2$ \nspectral binning {} channels'.format(epoch, spatial_binning_factor, spatial_binning_factor, spectral_binning_factor)

    print(text)
        
    #ax.text(0.7,0.85,text, transform=ax.transAxes)
    ax.text(0.15,0.7,text, transform=ax.transAxes)#"", fontsize=13)
    
    ax.errorbar(X, Y_Wm2, Y_err_Wm2,
                marker=MARKERS_IFS[4],label='', color='c')

    mise_en_page(ax,x_step=1, x_maj_step=0.1, x_min_step=0.01, y_step=0, y_maj_step=0.5, y_min_step=0.1)
    ylim = ax.get_ylim()
    ylim = [0,ylim[1]]
    ylim = [0,5e-17]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.legend(loc='best', numpoints=1, ncol=2, frameon=False)

    plt.savefig(saving_dir+namesave)
    if plot_show: plt.show()
    plt.clf()

    
    ## Save Dataframe
    dF = pd.DataFrame({'wavelength (mic)': X, 'flux_star (W/m^2)': Y_star_syn,
                           'flux (W/m^2)': Y_Wm2, 'flux_e (W/m^2)': Y_err_Wm2,
                           'contrast': Y_contr, 'contrast_e': Y_err_contr,
                           'flux (ADU/arcsec^2)': Y_ADUarcsec2, 'flux_e (ADU/arcsec^2)': Y_err_ADUarcsec2
                           })
    
    namesave='table_spectrum_{}_{}_mask={}_spatial_binning={}x{}pix_spectral_binning={}.csv'.format(epoch, instru, mask_type,
                    spatial_binning_factor, spatial_binning_factor,  spectral_binning_factor)
    dF.to_csv(saving_dir+namesave, sep=',')
