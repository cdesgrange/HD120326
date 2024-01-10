
from import_packages_generic import *
from import_functions_generic import *
from matplotlib.colors import LogNorm
from astropy import stats


## Block 1: Make Grids in Position Angle, Radius ##
def compute_im_pa_grid(im, xc, yc):
    '''
    Compute a 2D-array with position angle from the center of the image.

    Inputs:
        .'im' (2D-array): image
        .'xc'
        .'yc'
    Output:
        .'im_deg' (2d-array): image with values corresponding to the
        distance of the center (in pixels)
    '''

    ## most relevant: angle computation ##
    h, w = np.shape(im)
    deg_grid = np.zeros((h, w))
    for ix in range(w):
        for iy in range(h):
            x,y = xc-ix, yc-iy
            r = np.sqrt(x**2+y**2)
            theta = 2*np.arctan( y/(x+r)) + np.pi
            if np.isnan(theta): theta = 0
            deg_grid[iy,ix] = theta*180/np.pi

    return deg_grid


def compute_im_rad_grid(im, center='center', even_or_odd='even'):
    '''
    Compute a 2D-array with distance from the center of the image.

    Works if the image is even or odd.
    Do not look into the code, it is messy, very messy.
    Friday late afternoon...

    Inputs:
        .'im' (2D-array): image
        .'center' (string):
            Default value: center='center'.
            Other possible values:
                .if the image is even: 'n//2-1' or 'n//2'
                .otherwise, if the image is odd: 'n//2-1' or 'n//2+1'
                Assumption: Count starts to 0
        .'even_or_odd' (string):
            Default value: even_or_odd='even'
    Output:
        .'im_rad' (2d-array): image with values corresponding to the
        distance of the center (in pixels)
    '''

    h, w = np.shape(im)
    if center == 'center':
        x, y = np.arange(-w//2,w//2+1,1), np.arange(-h//2,h//2+1,1)
        x = np.where(w%2==0, np.delete(x,w//2), np.delete(x,0))
        y = np.where(h%2==0, np.delete(y,h//2), np.delete(y,0))
        him, wim = h, w
    else :
        x, y = np.arange(-w//2,w//2+2,1), np.arange(-h//2,h//2+2,1)
        him, wim = h+1, w+1

    radius_grid = np.zeros((him, wim))
    for ix in range(wim):
        for iy in range(him):
            radius_grid[iy,ix] = x[ix]**2 + y[iy]**2

    if even_or_odd == 'even':
        if center == 'n//2-1' : radius_grid = radius_grid[1:,1:]
        elif center == 'n//2' : radius_grid = radius_grid[:-1,:-1]
        elif center != 'center' : return print("Careful, check the value assigned to the variable 'center'.")

    if center != 'center' and even_or_odd == 'odd':
        if center == 'n//2-1':
            im_rad = compute_im_rad_grid(np.zeros((h+1,w+1)), center = center, even_or_odd = 'even')
            im_rad = im_rad[1:,1:]
        elif center == 'n//2+1':
            center = 'n//2'
            im_rad = compute_im_rad_grid(np.zeros((h+1,w+1)), center = center, even_or_odd = 'even')
            im_rad = im_rad[:-1,:-1]
        else:  print("Careful, check the value assigned to the variable 'center'.")
        return im_rad

    return np.sqrt(radius_grid)



## Block 2: Compute detection limits by slippy box, annulus, section of annulus, slippy section of annulus ##
def compute_limdet_im_box_slippy(res_map, dx, dy):
    '''
    Compute the detection limit 2D map by slippy box estimation of the noise.

    Inputs:
        .'res_map' (2D-array): residual map
        .'dx' (int): width of the box on which the noise is computed
        .'dy' (int): height of the box on which the noise is computed

    Output:
        .'im_noise' (2D-array): detection limit map

    Warning: The height and width of the image should be of identic.
    '''
    print("\nComputing the detection limit map by using the slippy box method")
    t0 = time.time()
    w,h = np.shape(res_map)
    im_noise = np.zeros((w,h))
    if w != h : print('The height and width of the image should be of identic. Here:', h,w)
    X = np.arange(w)
    XX,YY = np.meshgrid(X,X)
    for ix in range(w):
        print('Percentage done {:.1f} %'.format(100*ix/w))
        for iy in range(h):
            cond_x = np.abs(ix-XX) <= dx
            cond_y = np.abs(iy-YY) <= dy
            cond_box = np.logical_and(cond_x,cond_y)
            noise = np.nanstd(res_map[cond_box])
            im_noise[iy,ix] = noise
    print("Took {} seconds".format(time.time()-t0))
    return im_noise


def compute_limdet_map_ann(res_map, dr, alpha=2, center='center', even_or_odd='even'):
    '''
    Compute the detection limit 2D map by annular noise estimation with slippery annulus.
    Use the function compute im_rad_grid() to derive the distance (in pixel) of each
    pixel to the center of the image.

    Inputs:
        .'res_map' (2D-array): residual map
        .'dr' (float): width of the annulus for which the detection limit is computed
        (optional)
        .'alpha' (float): factor to consider bigger annulus to derive the noise
            Goal: have smoother transition between annuli
        .'center' (string):
            Default value: center='center'.
            Other possible values:
                .if the image is even: 'n//2-1' or 'n//2'
                .otherwise, if the image is odd: 'n//2-1' or 'n//2+1'
                Assumption: Count starts to 0
        .'even_or_odd' (string):
            Default value: even_or_odd='even'

    Output:
        .'im_noise' (2D-array): detection limit map
    '''
    t0 = time.time()
    print("\nComputing the detection limit map by using the 1D-annulus method")
    h, w = np.shape(res_map)
    #res_map_pos = np.copy(res_map)
    #res_map_pos[res_map_pos<0] = 0
    noise_tot, noise_ann = [], []

    im_noise, im_nan, im_radius_grid = np.zeros((h,w)), np.empty((h,w)), compute_im_rad_grid(res_map)
    im_nan[:] = np.NaN
    rad, x0,y0 = alpha, w//2+1, h//2+1

    # Until the annulus is smaller than the size of the image
    while rad < w//2 - alpha-dr :
        # One annulus is considered
        rad += dr
        cond_annulus_large = np.logical_and(im_radius_grid >= rad-alpha, rad + dr + alpha >= im_radius_grid)
        cond_annulus_thin  = np.logical_and(im_radius_grid >= rad, rad + dr >= im_radius_grid)
        im_annulus = np.where(cond_annulus_large, res_map, im_nan)
        # the noise over the annulus is computed
        noise_annulus = np.nanstd(im_annulus)
        # and the image is set at this noise for this given annulus
        im_noise[cond_annulus_thin] = noise_annulus
    print("Took {} seconds".format(time.time()-t0))
    return im_noise


def compute_limdet_map_ann_sec(res_map, nb_pixel_sec, dr, alpha=2, center='center', even_or_odd='even'):
    '''
    Compute the detection limit 2D map by annular noise estimation with slippery annulus.
    Use the function compute im_rad_grid() to derive the distance (in pixel) of each
    pixel to the center of the image.

    Inputs:
        .'res_map' (2D-array): residual map
        .'dr' (float): width of the annulus for which the detection limit is computed
        (optional)
        .'alpha' (float): factor to consider bigger annulus to derive the noise
            Goal: have smoother transition between annuli
        .'center' (string):
            Default value: center='center'.
            Other possible values:
                .if the image is even: 'n//2-1' or 'n//2'
                .otherwise, if the image is odd: 'n//2-1' or 'n//2+1'
                Assumption: Count starts to 0
        .'even_or_odd' (string):
            Default value: even_or_odd='even'

    Output:
        .'im_noise' (2D-array): detection limit map
    '''
    t0 = time.time()
    print("\nComputing the detection limit map by using the annulus + section method")

    h, w = np.shape(res_map)
    #res_map_pos = np.copy(res_map)
    #res_map_pos[res_map_pos<0] = 0
    noise_tot, noise_ann = [], []

    im_noise, im_nan, im_radius_grid = np.zeros((h,w)), np.empty((h,w)), compute_im_rad_grid(res_map)
    im_nan[:] = np.NaN
    rad, x0,y0 = alpha, w//2+1, h//2+1

    if center == 'n//2': xc, yc = w//2, h//2
    else : xc, yc = w/2, h/2

    im_pa_grid = compute_im_pa_grid(res_map, xc, yc)

    # Until the annulus is smaller than the size of the image
    while rad < w//2 - alpha-dr :
        # One annulus is considered
        rad += dr
        cond_annulus_large = np.logical_and(im_radius_grid >= rad-alpha, rad + dr + alpha >= im_radius_grid)
        cond_annulus_thin  = np.logical_and(im_radius_grid >= rad, rad + dr >= im_radius_grid)
        #im_annulus    = np.where(cond_annulus_large, res_map_pos, im_nan)

        nb_pixel_ann = np.nansum( np.where(cond_annulus_thin, 1, np.nan) )
        nb_sections  = nb_pixel_ann//nb_pixel_sec
        dpa =  360/nb_sections
        pa = dpa/2
        print("Radius at {} pixels, considers {} sections".format(rad,nb_sections))

        while pa < 360 + dpa/2 :
            cond_sec_large = np.logical_and(im_pa_grid >= pa-dpa/2-alpha, pa + dpa/2 + alpha >= im_pa_grid)
            cond_sec_thin  = np.logical_and(im_pa_grid >= pa-dpa/2, pa + dpa/2 >= im_pa_grid)

            # the noise over the section of the annulus is computed
            im_annulus_sec    = np.where(np.logical_and(cond_annulus_large, cond_sec_large), res_map, im_nan)
            noise_annulus_sec = np.nanstd(im_annulus_sec)
            #print(noise_annulus_sec)
            #np.nanmedian(im_annulus_sec[im_annulus_sec!=0])

            # and the image is set at this noise for this given annulus and section
            im_noise[np.logical_and(cond_annulus_thin,cond_sec_thin)] = noise_annulus_sec
            pa += dpa
    print("Took {} seconds".format(time.time()-t0))
    return im_noise



def compute_limdet_map_ann_sec_slippy(res_map, nb_pixel_sec, dr, center='center', even_or_odd='even', crop_owa=None):
    '''
    Compute the detection limit 2D map by annular noise estimation with slippery annulus.
    Use the function compute im_rad_grid() to derive the distance (in pixel) of each
    pixel to the center of the image.

    Inputs:
        .'res_map' (2D-array): residual map
        .'dr' (float): width of the annulus for which the detection limit is computed
        (optional)

        .'center' (string):
            Default value: center='center'.
            Other possible values:
                .if the image is even: 'n//2-1' or 'n//2'
                .otherwise, if the image is odd: 'n//2-1' or 'n//2+1'
                Assumption: Count starts to 0
        .'even_or_odd' (string):
            Default value: even_or_odd='even'

    Output:
        .'im_noise' (2D-array): detection limit map
    '''
    print("\nComputing the detection limit map by using the annulus + slippy sections method")
    t0 = time.time()
    h, w = np.shape(res_map)
    #res_map_pos = np.copy(res_map)
    #res_map_pos[res_map_pos<0] = 0
    noise_tot, noise_ann = [], []

    im_noise, im_nan, im_radius_grid = np.zeros((h,w)), np.empty((h,w)), compute_im_rad_grid(res_map)
    im_nan[:] = np.NaN

    if center == 'n//2': xc, yc = w//2, h//2
    else : xc, yc = w/2, h/2
    print('Compute the position angle grid')
    im_pa_grid = compute_im_pa_grid(res_map, xc, yc)

    crop = np.where(crop_owa == None, 0, crop_owa)

    for ix in range(crop, w-crop):
        print('Percentage {:.1f} %'.format(100*ix/w))
        for iy in range(crop, h-crop):
            x,y = xc-ix, yc-iy
            rad = np.sqrt(x**2+y**2)
            pa = im_pa_grid[iy,ix]
            cond_annulus_large = np.logical_and(im_radius_grid >= rad-dr, rad + dr  >= im_radius_grid)

            # find the right dpa based on the requirement nb_pixel_sec
            nb_pixel_ann = np.nansum( np.where(cond_annulus_large, 1, np.nan) )
            nb_sections  = nb_pixel_ann//nb_pixel_sec
            dpa =  360/nb_sections

            cond_pa_large = np.logical_and(im_pa_grid >= pa-dpa/2, pa + dpa/2 >= im_pa_grid)
            cond_sec   = np.logical_and(cond_annulus_large, cond_pa_large)
            im_ann_sec = np.where(cond_sec, res_map, im_nan)
            im_noise[iy,ix] = np.nanstd(im_ann_sec)
    print("Took {} seconds".format(time.time()-t0))
    print("Took {} minutes".format((time.time()-t0)/60))
    print("Took {} hours".format((time.time()-t0)/3600))
    return im_noise



## Block 3: Plotting ##
def plot_map(map, label, vmin = 1e-8, vmax=1e-5):
    fig,ax = plt.subplots(1,1,figsize=(7,6))
    fig.subplots_adjust(left=0.05,right=0.8, top = 0.92, bottom=0.05)

    ax.axis('off')
    im = plt.imshow(map,origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='nearest', norm=LogNorm(vmin, vmax))
    cax = plt.axes([0.85, 0.1, 0.045, 0.8])
    fig.colorbar(im,cax=cax)
    ax.text(0, np.shape(map)[0],label)
    scalebar = AnchoredSizeBar(ax.transData, 1000/12.25, '1"', 4, label_top=True, pad=0.1, sep=2,
                               borderpad=0.1, frameon=False, size_vertical=2, color='black') #,fontproperties=fp)
    ax.add_artist(scalebar)
    plt.savefig("AU_Mic_{}.pdf".format(label.replace(' ','_')))
    plt.show()
    return

def plot_comparison_contrast_curves(IM, LAB, LS=[':','--','-.','-','-.'], pixarc=12.25):
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    fig.subplots_adjust(left=0.15,right=0.95, top = 0.95, bottom=0.15)

    for k in range(len(IM)):
        im = IM[k]
        m = np.shape(im)[0]//2

        X = np.arange(0,m,1)*pixarc*1e-3
        Y = stats.circstats.circmean(im, axis=1)
        Y = Y[m:]
        plt.semilogy(X,Y, label = LAB[k], ls = LS[k])

    plt.legend(loc=0, frameon = False)

    ax.set_ylim([1e-8,3e-4])
    ax.set_xlim([1e-1,6])

    ax.fill_betweenx([1e-8,1e-7],1e-1,10,ls='-',color=[0.9,0.9,0.9],label='IWA')

    ax.set_xlabel('Separation (")')
    ax.set_ylabel('Contrast ($5\sigma$)')
    mise_en_page(ax)
    plt.savefig('contrast_curve_comparison.pdf')
    plt.show()
    return
