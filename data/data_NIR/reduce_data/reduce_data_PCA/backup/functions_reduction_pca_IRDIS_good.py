from import_packages_generic import *
from import_functions_generic import *
import vip_hci as vip



def load_data(data_dir, band, epoch, instru, channels='first', crop=None, sortframe=None, saving_dir=None, label='', namesave=None):
    instru_lab = np.where(instru=='IRDIS','ird','ifs')
    path_convert = '{}convert/{}/*{}_{}_convert*/'.format(data_dir,instru,epoch,np.where(instru=='IRDIS', 'ird', 'ifs'))
    print("Data are looked at ", path_convert)
    cube, header = fits.getdata(glob( path_convert + '*center_im*' )[0],header=True)
    psf  = fits.getdata(glob( path_convert + '*median_unsat*' )[0])
    pa   = - fits.getdata(glob( path_convert + '*rotnth*' )[0]) # SPHERE DC convention v.s. VIP convention
    lbd  = fits.getdata(glob( path_convert + '*lam*' )[0])

    #if band == 'BB_H' :

    print("Shape of the original cube", np.shape(cube))
    print("Shape of the original psf", np.shape(psf))
    print("Shape of the original pa", np.shape(pa))
    print("Shape of the original lbd", np.shape(lbd))

    #if instru == 'IRDIS' and band not in ['H23','J23','J','H','K'] : # band = H2 or H3
        # add case BB_H
    #    print("The band considered is ",band)
    #    cube = np.where(band in ['J2','H2','K1'], cube[0], cube[1])
    #    psf  = np.where(band in ['J2','H2','K1'], psf[0],  psf[1])
    #    lbd  = np.where(band in ['J2','H2','K1'], lbd[0] , lbd[1])

    # add case several psf

    if sortframe != None and sortframe != 'None':
        path_sortframe = '{}sortframes/*{}_{}_sortf*/'.format(data_dir,epoch,np.where(instru=='IRDIS', 'ird', 'ifs'))
        print("Sortframe files are looked at ", path_sortframe)
        # Sortframes files from SPHERE-DC
        sortframe_specal, header = fits.getdata(glob( path_sortframe + '*frame_selection*' )[0],header=True)
        flag = np.array(sortframe_specal[:, np.where(sortframe == 'soft', 1, 0)])
        flag = np.where(flag==1,True,False)

        print("Good frames from SpeCal equal True\n", flag)

        # Sortframe files defined a la mano
        try :
            dF_sortframe_homemade = pd.read_csv( glob( path_sortframe + '*badfram*' )[0], comment ='#')
            sortframe_homemade = np.array(dF_sortframe_homemade)
            print("Indexes of the bad frames from homemade routine\n", sortframe_homemade)
            flag[sortframe_homemade] = False
            print("Good frames from SpeCal+homemade equal True\n", flag)
        except : print("No frames removed a la mano")

        print("Remove {} frames ({} sorting).".format(len(flag[flag==False]), sortframe))

        cube, pa = cube[:, flag], pa[flag]

        if saving_dir != None :
            os.makedirs(saving_dir+'fits/',exist_ok=True)
            fits.writeto(saving_dir+'fits/{}_center_im.fits'.format(namesave), cube, overwrite=True)

    if crop != None :
        print("Crop data")
        cube = cube[:,:,crop:-crop,crop:-crop]

    if channels != 'both': # thus channels = either 'first' or 'second'
        idx_good_channel = np.where(channels=='first', 0, 1)
        cube, psf, lbd = cube[idx_good_channel], psf[idx_good_channel], lbd[idx_good_channel]


    print("After rearranging the data:")
    print("Shape of the cube", np.shape(cube))
    print("Shape of the psf", np.shape(psf))
    print("Shape of the pa", np.shape(pa))
    print("Shape of the lbd", np.shape(lbd))

    return cube, header, psf, pa, lbd



def run_and_collapse_pca(cube, pa, nmodes, psf, header=None, technique = 'combine_before', mask_rad = 6, platescale=12.25, save_fits=1,
                    plot_modes = 1, plot_res_maps = 1, saving_dir = None, namesave = None,
                    modes_vminmax = 1, modes_vmin = 0, modes_vmax = 1e-2,
                    res_vminmax = 1, res_vmin = 0, res_vmax = 1e-1):

        normalization_factor = np.array([normalization(psf[i]) for i in range(2)])

        print("The factor of normalization are:", normalization_factor)

        if technique == 'combine_after' :
            cube_im_pca, pcs = [], []
            for i in range(2):
                cube_1channel = cube[i]/normalization_factor[i]
                print("Shape of the cube in 1 channel:", np.shape(cube_1channel))
                print("Median of the cube in 1 channel:", np.nanmedian(cube_1channel))
                print("Mean of the cube in 1 channel:", np.nanmean(cube_1channel))
                print("Sum of the cube in 1 channel:", np.nansum(cube_1channel))
                cube_im_pca_1channel, pcs_1channel = run_pca(cube_1channel, pa, nmodes, header=header, mask_rad = mask_rad,
                                save_fits=save_fits, plot_modes = plot_modes, plot_res_maps = plot_res_maps,
                                saving_dir = saving_dir, namesave = namesave+'_channel'+str(i+1),
                                modes_vminmax = modes_vminmax, modes_vmin = modes_vmin, modes_vmax = modes_vmax,
                                res_vminmax = res_vminmax, res_vmin = res_vmin, res_vmax = res_vmax)
                cube_im_pca.append(cube_im_pca_1channel)
                pcs.append(pcs_1channel)

            cube_im_pca = np.nanmean( np.array(cube_im_pca), axis=0)
            pcs = np.nanmean( np.array(pcs), axis=0)

            plot_pca_modes(pcs, nmodes, saving_dir, namesave=namesave,
                        vminmax = modes_vminmax, vmin = modes_vmin, vmax = modes_vmax,
                        mask_rad=mask_rad, platescale=platescale)
            plot_pca_res_map(cube_im_pca, nmodes, saving_dir, namesave = namesave,
                        vminmax = res_vminmax, vmin = res_vmin, vmax = res_vmax,
                        mask_rad=mask_rad, platescale=platescale)

        else : # combine_before
            print("Collapse data before running the PCA.")
            cube[0] = cube[0]/normalization_factor[0]
            cube[1] = cube[1]/normalization_factor[1]
            print("Shape of the cube in 1 channel:", np.shape(cube[0]), np.shape(cube[1]))
            print("Median of the cube in 1 channel:", np.nanmedian(cube[0]), np.nanmedian(cube[1]))
            print("Mean of the cube in 1 channel:", np.nanmean(cube[0]), np.nanmean(cube[1]))
            print("Sum of the cube in 1 channel:", np.nansum(cube[0]), np.nansum(cube[1]))

            cube_collapse = np.nanmean(cube, axis=0)

            print("\nShape of the cube after collapse:", np.shape(cube_collapse))
            print("Median of the cube after collapse:", np.nanmedian(cube_collapse))
            print("Mean of the cube after collapse:", np.nanmean(cube_collapse))
            print("Sum of the cube after collapse:", np.nansum(cube_collapse))

            cube_im_pca, pcs = run_pca(cube_collapse, pa, nmodes, header=header, mask_rad = mask_rad,
                            save_fits=save_fits, plot_modes = plot_modes, plot_res_maps = plot_res_maps,
                            saving_dir = saving_dir, namesave = namesave,
                            modes_vminmax = modes_vminmax, modes_vmin = modes_vmin, modes_vmax = modes_vmax,
                            res_vminmax = res_vminmax, res_vmin = res_vmin, res_vmax = res_vmax)

        print("\nShape of the cube after collapse and PCA:", np.shape(cube_im_pca))
        print("Median of the cube after collapse and PCA:", np.nanmedian(cube_im_pca))
        print("Mean of the cube after collapse and PCA:", np.nanmean(cube_im_pca))
        print("Sum of the cube after collapse and PCA:", np.nansum(cube_im_pca))

        if save_fits :
            fits.writeto('{}fits/{}_cube_im_pca.fits'.format(saving_dir, namesave), cube_im_pca, header, overwrite=True)
            fits.writeto('{}fits/{}_cube_pcs.fits'.format(saving_dir, namesave), pcs, header, overwrite=True)

        return cube_im_pca, pcs



def run_pca(cube, pa, NMODES, header=None, mask_rad = None, platescale=12.25, save_fits = 1,
        plot_modes=1, plot_res_maps=1, saving_dir=None, namesave='',
        modes_vminmax = 1, modes_vmin = 0, modes_vmax = 1e-2,
        res_vminmax = 1, res_vmin = 0, res_vmax = 1e-1):

    cube_im_pca, cube_res_cube, cube_res_cube_derot = [], [], []
    for i in range(len(NMODES)):
        nmodes = NMODES[i]
        print("PCA running for nmodes = ", nmodes)
        im_pca, pcs, reconstr_cube, res_cube, res_cube_derot = vip.psfsub.pca_fullfr.pca(cube, pa, ncomp=nmodes, mask_center_px=mask_rad, full_output=True)
        cube_im_pca.append(im_pca)
        #cube_res_cube.append(res_cube)
        cube_res_cube_derot.append(res_cube_derot)

    cube_im_pca = np.array(cube_im_pca)   # reduced PCA  image
    cube_res_cube_derot = np.array(cube_res_cube_derot) # cube reduced PCA image already derotated -> Forward Modelling (?)

    if save_fits and saving_dir != None :
        print("\nWrite the cube of reduced PCA images for different number of components")
        print("at :", saving_dir+'fits/'+namesave+'.fits')
        fits.writeto(saving_dir+'fits/'+namesave+'_cube_im_pca'+'.fits', cube_im_pca, header, overwrite=True)
        fits.writeto(saving_dir+'fits/'+namesave+'_cube_pcs'+'.fits', pcs, header, overwrite=True)

    if plot_modes and saving_dir != None :
        plot_pca_modes(pcs, NMODES, saving_dir, namesave=namesave,
                    vminmax = modes_vminmax, vmin = modes_vmin, vmax = modes_vmax, mask_rad=mask_rad, platescale=platescale)

    if plot_res_maps and saving_dir != None :
        plot_pca_res_map(cube_im_pca, NMODES, saving_dir, namesave=namesave,
                    vminmax = res_vminmax, vmin = res_vmin, vmax = res_vmax, mask_rad=mask_rad, platescale=platescale)

    return cube_im_pca, pcs


def plot_pca_modes(pcs, NMODES, saving_dir=None, namesave='', vminmax = 1, vmin = 0, vmax = 1e-2, add_crop=240, platescale=12.25, mask_rad=7):
    ''' plot the components used to compute the PCA fullframe residuals map 'im_pca_fullfr' '''

    n = len(NMODES)
    nli, count = 2, -1
    nco = int(np.where( n//nli == int(n/nli), n//nli, n//nli+1))
    fig,ax = plt.subplots(nli,nco,figsize=(3*nco,3*nli))
    fig.subplots_adjust(left=0.05,right=0.8, top = 0.85, bottom=0.05, hspace=0.1, wspace=0.05)
    fig.suptitle('First modes of the PCA\n' + namesave.replace('_',' '), fontsize=16)

    for li in range(nli):
        for co in range(nco):
            count += 1
            ax[li,co].axis('off')
            if count < n :
                nmodes = NMODES[count]
                if add_crop != None : im = pcs[count][add_crop:-add_crop,add_crop:-add_crop]
                else : im = pcs[count]
                if vminmax : plot_im = ax[li,co].imshow(im,vmin=vmin,vmax=vmax,origin='lower')
                else : plot_im = ax[li,co].imshow(im,vmin=0,origin='lower')
                ax[li,co].set_title('mode {}'.format(count+1))

                # add scalebar
                scalebar = AnchoredSizeBar(ax[li,co].transData, 200/platescale, '0.2"', 4, label_top=True, pad=0.2, sep=2,
                                           borderpad=0.2, frameon=False, size_vertical=1.5, color='white') #,fontproperties=fp)
                ax[li,co].add_artist(scalebar)

                # add IWA
                circle = Circle((np.shape(im)[0]/2,np.shape(im)[0]/2),mask_rad/2,ec='black',fc='black',ls='-',lw=1)
                ax[li,co].add_patch(circle)

    if vminmax :
        cax = plt.axes([0.84, 0.05, 0.025, 0.8])
        cbar = fig.colorbar(plot_im,cax=cax)
        cbar.set_label('Contrast')

    if saving_dir != None:
        fn_save = saving_dir+'fig/'+namesave.replace(' ','_')+'_cube_pcs'
        os.makedirs(saving_dir+'fig/',exist_ok=True)
        plt.savefig(fn_save+'.pdf')
        plt.savefig(fn_save+'.png', dpi=500)
    plt.show()
    return

def plot_pca_res_map(cube_im_pca, NMODES, saving_dir=None, namesave='', vminmax = 1, vmin = 0, vmax = 1e-1, add_crop=240, platescale=12.25, mask_rad=7):
    # plot the PCA fullframe residuals map 'im_pca' for different number of components (1 to ncomp_max)
    n = len(NMODES)
    nli, count = 2, -1
    nco = int(np.where( n//nli == int(n/nli), n//nli, n//nli+1))
    fig,ax = plt.subplots(nli,nco,figsize=(3*nco,3*nli))
    fig.suptitle('PCA reduced maps for different number of modes\n' + namesave.replace('_',' '), fontsize=16)
    fig.subplots_adjust(left=0.05,right=0.8, top = 0.85, bottom=0.05, hspace=0.1, wspace=0.05)

    for li in range(nli):
        for co in range(nco):
            count += 1
            ax[li,co].axis('off')
            if count < n :
                nmodes = NMODES[count]
                if add_crop != None : im = cube_im_pca[count][add_crop:-add_crop,add_crop:-add_crop]
                else : im = cube_im_pca[count]
                if vminmax : plot_im = ax[li,co].imshow(im,vmin=vmin,vmax=vmax,origin='lower')
                else : plot_im = ax[li,co].imshow(im,vmin=0,origin='lower')

                ax[li,co].set_title('Nmodes = {}'.format(nmodes))

                # add scalebar
                scalebar = AnchoredSizeBar(ax[li,co].transData, 200/platescale, '0.2"', 4, label_top=True, pad=0.2, sep=2,
                                           borderpad=0.2, frameon=False, size_vertical=1.5, color='white') #,fontproperties=fp)
                ax[li,co].add_artist(scalebar)

                # add IWA
                circle = Circle((np.shape(im)[0]/2,np.shape(im)[0]/2),mask_rad/2,ec='black',fc='black',ls='-',lw=1)
                ax[li,co].add_patch(circle)

    if vminmax :
        cax = plt.axes([0.84, 0.05, 0.025, 0.8])
        cbar = fig.colorbar(plot_im,cax=cax)
        cbar.set_label('Contrast')

    if saving_dir != None:
        fn_save = saving_dir+'fig/'+namesave.replace(' ','_')+'_cube_im_pca'
        os.makedirs(saving_dir+'fig/',exist_ok=True)
        plt.savefig(fn_save+'.pdf')
        plt.savefig(fn_save+'.png',dpi=500)
    plt.show()
    return



def normalization(psf):
    factor_normalization = np.nansum(psf)
    print(factor_normalization)
    return factor_normalization
