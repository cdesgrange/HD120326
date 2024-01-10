#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:12:28 2023

@author: millij
"""
from pathlib import Path
import numpy as np
from betadisk import BetaDisk
import vip_hci as vip
ds9=vip.Ds9Window(wait=15)
import matplotlib.pyplot as plt
from astropy.io import fits

path = Path('/Users/millij/Documents/ScoCen_program/HD120326_ScoCen_double_belt/interpretation_beta_disk')

disk = BetaDisk(nx=300,ng=10,nl=int(2e7))

# # optional parameters for BetaDisk
# nx = 300           # [int] number of pixels for the images
# nl = 10_000_000    # [int] number of particles to be launched
# ng = 10            # [int] number of grain size intervals
# pixscale = 0.01226 # [float] size of one pixel in arcsec
# bmin = 0.001       # [float] minimum value for beta
# bmax = 0.49        # [float] maximum value for beta
# nb = 50            # [int] number of bins for the phase function (see later)
# slope = 0.5        # [float] "Correction" for the size distribution (see next paragraph)
# dx = 0.            # [float] possibility to have an offset in the x direction
# dy = 0.            # [float] and in the y direction


# # optional parameters for disk.compute_model
# a = 0.55              # [float] the reference radius of the disk, in arcsec
# pfunc = np.ones(nb)  # [np.array] array containing the phase function
# is_hg = True         # [boolean] should we use the HG approximation
# dpi = False          # [boolean] if is_hg is True, we can also model polarimetric observations

dstar = 107 #pc
a_au = 58.5
a_arcsec = a_au / dstar
print('Semi-major axis in arcsec',a_arcsec)

dr = a_arcsec/20           # [float] the standard deviation for the width of the main belt (normal profile)
incl = 80.           # [float] inclination of the disk, in degrees
pa = 90.           # [float] position angle of the disk, in degrees
opang = 0.05         # [float] opening angle of the disk
ghg = 0.8             # [float or np.array] value for the asymmetry parameter for the HG phase function

disk.compute_model(a = a_arcsec,dr = dr,incl=incl,pa=pa,opang = opang,\
                   is_hg=True,ghg=ghg,\
                   dpi = False )


print(np.shape(disk.model)) # (10, 300, 300)
ds9.display(disk.model)

disk_total_flux = np.sum(disk.model,axis=((1,2)))
beta_bin_edges = disk._bgrid
width_bins = np.diff(beta_bin_edges)
beta_bin_center = beta_bin_edges[:-1]+width_bins/2

disk_intensity = np.copy(disk.model)
disk_intensity_mean_over_beta = np.mean(disk_intensity,axis=0)


disk.compute_model(a = a_arcsec,dr = dr,incl=incl,pa=pa,opang = opang,\
                   is_hg=True,ghg=ghg,\
                   dpi = True )

disk_total_polarized_flux = np.sum(disk.model,axis=((1,2)))

disk_polarized_intensity = np.copy(disk.model)
disk_polarised_intensity_mean_over_beta = np.mean(disk_polarized_intensity,axis=0)

ds9.display(disk_intensity,disk_polarized_intensity)
ds9.display(disk_intensity_mean_over_beta,disk_polarised_intensity_mean_over_beta)


plt.close(1) # or better fig.clf() if it already exists
fig, ax = plt.subplots(1,1,figsize=(10,4),num=1)
ax.errorbar(beta_bin_center,disk_total_flux,xerr=width_bins/2,marker='o',linestyle=None,label='total intensity')
ax.errorbar(beta_bin_center,disk_total_polarized_flux,xerr=width_bins/2,marker='o',linestyle=None,label='polarised intensity')
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.legend(frameon=True,loc='best',fontsize=12)
ax.set_ylabel('Total flux (ADU)',fontsize=12)
ax.set_xlabel('$\\beta$ parameter',fontsize=12)
# ax.set_xlim(1e-1,1000)
# ax.set_ylim(15,10)
ax.grid()
fig.savefig(path.joinpath('flux_vs_beta.pdf'))

filename = 'a_{0:.0f}mas_dr_{1:.0f}mas_incl_{2:.0f}deg_ghg_{3:03.0f}_pa_{4:.0f}deg'.format(a_arcsec*1000,dr*1000,incl,ghg*100,pa)
fits.writeto(path.joinpath('total_intensity_{0:s}_cube.fits'.format(filename)),disk_intensity,overwrite=True)
fits.writeto(path.joinpath('polarised_intensity_{0:s}_cube.fits'.format(filename)),disk_polarized_intensity,overwrite=True)

fits.writeto(path.joinpath('total_intensity_{0:s}_mean.fits'.format(filename)),disk_intensity_mean_over_beta,overwrite=True)
fits.writeto(path.joinpath('polarised_intensity_{0:s}_mean.fits'.format(filename)),disk_polarised_intensity_mean_over_beta,overwrite=True)


# nx, ng = 1_000, 12
# ghg = np.linspace(0.9, 0.5, num = ng)
# disk = BetaDisk(nx = nx, ng = ng)
# disk.compute_model(a = 1.5, dr = 0.030, incl = 40.0, pa = -120.0, opang = 0.05, \
#                 is_hg = True, ghg = ghg)
# print(np.shape(disk.model))
# # Should return something like (12, 1000, 1000)

# r0_au = 58.6 
# d_pc = 107.4 # pc
# r0_arcsec = r0_au/d_pc
# print(r0_arcsec) # 0.545 arcsec
