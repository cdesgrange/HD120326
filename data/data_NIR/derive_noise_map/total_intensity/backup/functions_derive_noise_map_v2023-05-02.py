from import_functions_generic import *

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


def compute_limdet_map_ann(res_map, dr, alpha=2, center='n//2', even_or_odd='even', display=1):
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
            Default value: center='n//2'.
            Other possible values:
                .if the image is even: 'n//2-0.5' or 'n//2' YES
                .otherwise, if the image is odd: 'n//2-1' or 'n//2+1' NO
                Assumption: Count starts to 0
        .'even_or_odd' (string):
            Default value: even_or_odd='even'

    Output:
        .'im_noise' (2D-array): detection limit map
    '''
    t0 = time.time()
    if display: print("\nComputing the detection limit map by using the 1D-annulus method")
    h, w = np.shape(res_map)
    #res_map_pos = np.copy(res_map)
    #res_map_pos[res_map_pos<0] = 0
    noise_tot, noise_ann = [], []

    im_noise, im_nan, im_radius_grid = np.zeros((h,w)), np.empty((h,w)), compute_im_rad_grid(res_map, center=center)
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
    if display: print("Took {} seconds".format(time.time()-t0))
    return im_noise
