def compute_im_rad_grid(im, center='n//2', even_or_odd='even'):
    '''
    Compute a 2D radius grid with the same shape than the input image 'im'.
    Optional parameter 'even_or_odd' only considered as 'even'. (!)
    '''
    nx = np.shape(im)[1]
    ny = np.shape(im)[0]
    if center == 'n//2-0.5':
        x,y = np.linspace(-nx//2,nx//2,nx), np.linspace(-ny//2,ny//2,ny)
    elif center == 'n//2':
        x,y = np.arange(-nx//2,nx//2,1), np.arange(-ny//2,ny//2,1)
    else: raise ValueError('Check your value for the optional parameter center')
    xs,ys = np.meshgrid(x, y, sparse=True)
    zs = np.sqrt(xs**2 + ys**2)
    return zs

## Circle

def flux_center_circle(im, r):
    '''
    Return the total flux of the image 'im' below a radius 'r' given in pixels.
    '''
    im_rad = compute_im_rad_grid(im)
    mask = np.where(im_rad < r, True, False)
    flux = np.nansum(im[mask])
    return flux

def im_masked_center(im, r):
    '''
    Return the image 'im' masked below a radius 'r' given in pixels.
    '''
    im_rad = compute_im_rad_grid(im)
    im_mask = np.where(im_rad < r, im, False)
    return im_mask


## Annulus

def flux_in_annulus(im, rin, rout):
    '''
    Return the sum flux of the image 'im' between the rings 'rin' and 'rout' given in pixels.
    '''
    im_rad = compute_im_rad_grid(im)
    mask = np.logical_and( np.where(im_rad < rout, True, False), np.where(im_rad > rin, True, False) )
    flux = np.nansum(im[mask])
    return flux


def im_masked_annulus(im, rin, rout):
    '''
    Return the sum flux of the image 'im' between the rings 'rin' and 'rout' given in pixels.
    '''
    im_rad = compute_im_rad_grid(im)
    mask = np.logical_and( np.where(im_rad < rout, True, False), np.where(im_rad > rin, True, False) )
    im[~mask] = 0
    return im


## Derive values and plot them

def shortcut_plot_flux_star_and_background(ax, im_i, r_star=4, r_bkg_in=25, r_bkg_out=30, 
                                           plot_bkg=1, plot_star=1, ms_bkg=8, ms_star=12, 
                                           color_bkg=[0,0,0.3], color_star='gold', do_norm=1, display=0):
    # Several channels
    nlbd = np.shape(im_i)[0]
    if display: print('Shape', np.shape(im_i))
    for ilbd in range(nlbd):
        im = np.copy(im_i[ilbd])
        #print(np.shape(im))
        n = np.shape(im)[0]//2; 
        # Estimate the flux of the Background and the Star 
        if plot_bkg:
            Y_bkg = flux_in_annulus(np.copy(im), r_bkg_in, r_bkg_out)
            if display: print('\n- Flux background total: %.3e' % Y_bkg)
            if do_norm :
                im_mask = im_masked_annulus(np.copy(im), r_bkg_in, r_bkg_out)
                im_count = np.copy(im_mask)
                im_count[im_count!=0]=1
                nb_pix = np.nansum(im_count)
                if display: print('Number of pixels considered:', nb_pix)
                Y_bkg /= nb_pix
                if display: print('Flux background / nb_pixels: %.3e' % Y_bkg)
                
            if color_bkg == None:
                ax.plot([ilbd+1],[Y_bkg], marker='o', ms=ms_bkg)
            else: ax.plot([ilbd+1],[Y_bkg], marker='o', ms=ms_bkg, color=color_bkg)
            
        if plot_star:
            Y_star = flux_center_circle(im,  r_star) 
            if display: print('\n- Flux star total: %.3e' % Y_star)
            if do_norm :
                im_mask = im_masked_center(im, r_star)
                im_count = np.copy(im_mask)
                im_count[im_count!=0]=1
                nb_pix = np.nansum(im_count)
                if display: print('Number of pixels considered:', nb_pix)
                Y_star /= nb_pix
                if display: print('- Flux star / nb_pixels: %.3e' % Y_star)
                    
            if color_star == None:
                ax.plot([ilbd+1],[Y_star], marker='*', ms=ms_star)
            else: ax.plot([ilbd+1],[Y_star], marker='*', ms=ms_star, color=color_star)
    return
