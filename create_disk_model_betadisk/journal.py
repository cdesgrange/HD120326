import os
from astropy.io import fits
from betadisk import *

from import_functions_generic import *

disk = BetaDisk()


def create_disk_model(# set-up
                      nx=300, nl=1000000, ng = 12, nb = 50,
                      pixscale = 0.01226, bmax = 0.49, bmin = 0.001, slope = 0.5, dx = 0., dy = 0,
                      # thermal image
                      thermal = False, lstar = None, dpc = None, wave = None,
                      # disk modelling
                      a=1.5, dr=0.25, incl=45, pa=-110, opang=0.05, is_hg=True, ghg = 0, dpi= False,
                      saving_dir='', namesave='', save_fits=1, plot=1):
    '''
    Inputs:
    # set-up
    nx = 300           # [int] number of pixels for the images
    nl = 10_000_000    # [int] number of particles to be launched
    ng = 10            # [int] number of grain size intervals
    nb = 50            # [int] number of bins for the phase function (see later)
    pixscale = 0.01226 # [float] size of one pixel in arcsec
    bmin = 0.001       # [float] minimum value for beta
    bmax = 0.49        # [float] maximum value for beta
    dx = 0.            # [float] possibility to have an offset in the x direction
    dy = 0.            # [float] and in the y direction
    
    slope = 0.5        # [float] "Correction" for the size distribution (see README)

    # for thermal images
    thermal = False    # [boolean] False by default, need to switch it to True to model e.g. ALMA data
    lstar = None       # [float] to compute the temperature we need a stellar luminosity
    dpc = None         # [float] we will also need a distance in pc (all distances in the code are in arcseconds)
    wave = None        # [float] and we need to provide a wavelength in microns

    # disk modelling
    a = 1.5              # [float] the reference radius of the disk, in arcsec
    dr = 0.25            # [float] the standard deviation for the width of the main belt (normal profile)
    incl = 45.           # [float] inclination of the disk, in degrees
    pa = -110.           # [float] position angle of the disk, in degrees
    opang = 0.05         # [float] opening angle of the disk
    pfunc = np.ones(nb)  # [np.array] array containing the phase function
    is_hg = True         # [boolean] should we use the HG approximation
    ghg = 0.             # [float or np.array] value for the asymmetry parameter for the HG phase function
    dpi = False          # [boolean] if is_hg is True, we can also model polarimetric observations

    '''
    
    if nx%2 ==0:
        cx = nx//2 - 0.5
    else:
        cx = nx//2
    disk = BetaDisk(nx = nx, ng = ng)

    bgrid = np.linspace(bmin, bmax, num = ng+1)

    print('The bgrid is:', np.round(bgrid,3))
    t0 = time.perf_counter()
    disk.compute_model(a=a, dr=dr, incl=incl, pa=pa, opang=opang, ghg=ghg,is_hg=is_hg,  dpi=dpi)
    print('Whole process took: {:.2f} sec'.format(time.perf_counter()-t0)) #dpi=false or true

    # disk model, beta_bin_edges
    disk_model = np.copy(disk.model)
    beta_bin_edges = disk._bgrid

    if save_fits:
        fits.writeto(saving_dir+namesave+'.fits',disk_model,overwrite=True)
        disk_mean = np.nanmean(disk_model, axis=0)
        fits.writeto(saving_dir+namesave+'_mean.fits',disk_mean,overwrite=True)
        

    if plot:
        """
        Make a pretty plot
        """
        nli=3
        nco=ng//nli
        xlim = cx * disk._pixscale
        
        fig = plt.figure(figsize=(10, 10*nli/nco))
        ax = plt.GridSpec(nli, nco, hspace=0.0, wspace=0.0)
        ax.update(left=0.0,right=1.,top=1.,bottom=0.0,wspace=0.0,hspace=0.00)
        ct = 0
        for i in range(nli):
            for j in range(nco):
                ax1 = fig.add_subplot(ax[i,j])
                ax1.imshow(disk.model[ct,], origin='lower', extent = [xlim, -xlim, -xlim, xlim],
                       vmax=np.percentile(disk.model[ct,], 99.5), cmap = 'inferno')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_axis_off()
                ct+=1
        
        if saving_dir != '': os.makedirs(saving_dir,exist_ok=True)
        plt.savefig(saving_dir+namesave+'.pdf')
        plt.show()
    return disk_model, beta_bin_edges


def plot_flux_vs_beta(disk_polarized, disk_total_intensity, beta_bin_edges, figsize=(6,3.5),
                    saving_dir='',prefix='', namesave='flux_vs_beta'):

    # beta bin edges
    width_bins = np.diff(beta_bin_edges)
    beta_bin_center = beta_bin_edges[:-1]+width_bins/2

    #print(np.shape(disk_total_polarized_flux), np.shape(disk_total_flux), np.shape(beta_bin_center))

    # total flux: sum over beta
    disk_total_flux = np.nansum(disk_total_intensity, axis=((1,2)))
    disk_total_polarized_flux = np.nansum(disk_polarized, axis=((1,2)))

    print(np.shape(disk_total_polarized_flux), np.shape(disk_total_flux), np.shape(beta_bin_center))

    
    #plt.close(1) # or better fig.clf() if it already exists
    fig, ax = plt.subplots(1,1,figsize=figsize,num=1)
    ax.errorbar(beta_bin_center,disk_total_flux,xerr=width_bins/2,marker='o',linestyle=None,label='total intensity', color='mediumblue')
    ax.errorbar(beta_bin_center,disk_total_polarized_flux,xerr=width_bins/2,
                marker='o',linestyle=None,label='polarised intensity', color='deeppink' )#'mediumvioletred')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.legend(frameon=True,loc='best',fontsize=12,numpoints=1)
    ax.set_ylabel('Total flux (ADU)',fontsize=12)
    ax.set_xlabel('$\\beta$ parameter',fontsize=12)
    # ax.set_xlim(1e-1,1000)
    # ax.set_ylim(15,10)
    #ax.grid()
    mise_en_page(ax)
    #fig.tight_layout()
    fig.savefig(saving_dir+prefix+namesave+'.pdf')




    
if __name__ == "__main__":
    L = time.localtime()
    date = "{}-{}-{}".format(L[0],L[1],L[2],L[3],L[4],L[5])

    saving_dir = 'figs/fig_v{}/'.format(date)
    os.makedirs(saving_dir,exist_ok=True)

    nx=300
    
    r = 0.55 # from Olofsson+2022b
    dr = 0.01
    pa = 86.4
    incl = 76.7
    ghg = 0.8
    ng=12

    plot=1

    do_it_one_given_g, do_it_several_g = 0,1

    ## For one given g
    if do_it_one_given_g:
        ghg_string = str(ghg)
        # dpi
        dpi=True
        namesave='HD120326_model_r={}_pa={}_incl={}_ghg={}_dr={}_dpi={}'.format(r, pa, incl, ghg_string, dr, dpi)
        disk_model_polar, beta_bin_edges = create_disk_model(nx=nx, a=r, pa=pa, incl=incl, ghg=ghg, ng=ng, dr=dr, dpi=dpi, plot=plot, saving_dir=saving_dir, namesave=namesave)

        # no dpi
        dpi=False

        namesave='HD120326_model_r={}_pa={}_incl={}_ghg={}_dr={}_dpi={}'.format(r, pa, incl, ghg_string, dr, dpi)
        disk_model_total, beta_bin_edges = create_disk_model(nx=nx, a=r, pa=pa, incl=incl, ghg=ghg, dr=dr, dpi=dpi, plot=plot, saving_dir=saving_dir, namesave=namesave)

        # Figure flux vs beta
        plot_flux_vs_beta(disk_model_polar, disk_model_total, beta_bin_edges,
                      saving_dir=saving_dir, prefix=namesave+'_')

        
    if do_it_several_g:
        ## For several g
        ghg = np.linspace(0.9, 0.5, num = ng)

        print('The ggrid is:', np.round(ghg,3))
        
        ghg_string= '{}-{}'.format(ghg[0], ghg[-1])

        # dpi
        dpi=True
        namesave='HD120326_model_r={}_pa={}_incl={}_ghg={}_dr={}_dpi={}'.format(r, pa, incl, ghg_string, dr, dpi)
        disk_model_polar, beta_bin_edges = create_disk_model(nx=nx, a=r, pa=pa, incl=incl, ghg=ghg, ng=ng, dr=dr, dpi=dpi, plot=plot, saving_dir=saving_dir, namesave=namesave)

        # no dpi
        dpi=False

        namesave='HD120326_model_r={}_pa={}_incl={}_ghg={}_dr={}_dpi={}'.format(r, pa, incl, ghg_string, dr, dpi)
        disk_model_total, beta_bin_edges = create_disk_model(nx=nx, a=r, pa=pa, incl=incl, ghg=ghg, dr=dr, dpi=dpi, plot=plot, saving_dir=saving_dir, namesave=namesave)

    
        # Figure flux vs beta
        plot_flux_vs_beta(disk_model_polar, disk_model_total, beta_bin_edges,
                      saving_dir=saving_dir, prefix=namesave+'_')

     
    
