from import_packages_generic import *
from import_functions_generic import *
import vip_hci as vip

from functions_reduction_pca import *
import yaml
import argparse

## General params for the simulations
data_dir   = 'fits/'
saving_dir = ''

path = 'fits/*cube_im_pca*'
flist = glob(path)
plot_res_maps, plot_modes = 1, 0
res_vminmax, res_vmin, res_vmax = 1, 0, 1e-7
modes_vminmax, modes_vmin, modes_vmax = 1, 0, 1e-3
platescale, mask_rad = 12.25, 7

print("flist:\n", flist)

for name in flist :
    print("\n=== Consider the reduction {} ===\n".format(name))
    namesave = name[5:-5]
    print("The namesave of the data is {}".format(namesave))

    data = fits.getdata(name)
    nmodes = np.arange(1,len(data)+1)

    print("Nmodes = ", nmodes)

    if plot_res_maps:
        print("Plot")
        try:
            plot_pca_res_map(data, nmodes, saving_dir, namesave=namesave,
                    vminmax = res_vminmax, vmin = res_vmin, vmax = res_vmax,
                    mask_rad=mask_rad, platescale=platescale)
        except:
            print('(!) Problem in plot_res_maps() (!)')
            pass

    if plot_modes:
        print("Plot")
        try:
            plot_pca_modes(data, nmodes, saving_dir, namesave=namesave,
                    vminmax = modes_vminmax, vmin = modes_vmin, vmax = modes_vmax,
                    mask_rad=mask_rad, platescale=platescale)
        except:
            print('(!) Problem in plot_pca_modes() (!)')
            pass
