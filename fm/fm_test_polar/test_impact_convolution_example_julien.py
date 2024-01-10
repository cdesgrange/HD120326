#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 22:46:40 2023

@author: millij
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from image_tools import angle_array
from scipy import signal

import vip_hci as vip
ds9 = vip.Ds9Window(wait=15)


path_input = Path('/Users/millij/Documents/HD181327_paper/HD181327_2017-05-16/analysis_polar/optimisation_polar_v2')
path_output = Path('/Users/millij/Documents/HD181327_paper/referee_answer')

path_root_intensity = Path('/Users/millij/Documents/HD181327_paper/HD181327_2017-05-16/')
    
fake_disk_map = fits.getdata(path_input.joinpath('model_best_least_square_final_iteration.fits'))
fake_disk_map = fake_disk_map/np.max(fake_disk_map)
psf = fits.getdata(path_root_intensity.joinpath('NewReduction/calibration/flux/HD181327_2017-05-16_master_flux.fits'))
psf_irdis_cropped = psf[512-16:512+16,512-16:512+16] 
psf_irdis_cropped = psf_irdis_cropped/np.sum(psf_irdis_cropped)
ds9.display(psf_irdis_cropped)

angarr = angle_array(fake_disk_map.shape)

ds9.display(fake_disk_map)

Q = -fake_disk_map*np.cos(2*angarr)
U = -fake_disk_map*np.sin(2*angarr)

ds9.display(Q,U)


fake_disk_convoled = signal.fftconvolve(fake_disk_map,psf_irdis_cropped, mode='same')

fake_disk_map_tmp_Q=fake_disk_map*np.cos(2*angarr)
fake_disk_map_tmp_U=fake_disk_map*np.sin(2*angarr)
fake_disk_map_tmp_Q_convolved = signal.fftconvolve(fake_disk_map_tmp_Q,psf_irdis_cropped, mode='same')
fake_disk_map_tmp_U_convolved = signal.fftconvolve(fake_disk_map_tmp_U,psf_irdis_cropped, mode='same')

fake_disk_map_convolved_rigourous = (fake_disk_map_tmp_Q_convolved*np.cos(2*angarr)+fake_disk_map_tmp_U_convolved*np.sin(2*angarr))
# fake_disk_map_tmp_pi_convolved = np.sqrt(fake_disk_map_tmp_Q_convolved**2+\
#                                          fake_disk_map_tmp_U_convolved**2)

ds9.display(fake_disk_map_tmp_Q_convolved,fake_disk_map_tmp_U_convolved,fake_disk_map_convolved_rigourous,fake_disk_convoled)


ds9.display(fake_disk_map_convolved_rigourous,fake_disk_convoled)
residuals = fake_disk_convoled-fake_disk_map_convolved_rigourous

relative_change = residuals/fake_disk_map_convolved_rigourous

ds9.display(fake_disk_map_convolved_rigourous,relative_change)

id_fainter_1percent = fake_disk_map_convolved_rigourous<0.01

relative_change_masked = np.copy(relative_change)
relative_change_masked[id_fainter_1percent] = np.nan

ds9.display(fake_disk_map_convolved_rigourous,relative_change_masked)

np.nanmax(np.abs(relative_change_masked))


ds9.display(fake_disk_convoled,fake_disk_map_tmp_pi_convolved,residuals)
# this shows that there is no change whether we derive Q and U and pI and then convolve
print(np.max(residuals)/np.max(fake_disk_convoled)) #0.0015
# ds9.display(Q_phi_frame_cropped,fake_disk_convoled)
