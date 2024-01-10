#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2024/01/10, based on Julien Milli's code
"""
Created on Tue Oct 31 22:46:40 2023

@author: millij
"""

from import_functions_generic import *
from functions_derive_noise_map import *

#import numpy as np
#from astropy.io import fits
#from pathlib import Path
#from image_tools import angle_array
from scipy import signal

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk

#import vip_hci as vip
#ds9 = vip.Ds9Window(wait=15)


print('=== Initialization ===')

save_files=1
scenario = '_HD120326' 
#scenario = '_Engler2018' 
#scenario = '_HeikampKeller2019_edge_on_far'

print('Do scenario', scenario[1:])

path_infolder = '/Users/desgranc/Documents/work/GitHub_clone/projects/HD102326/fm/fm_test_polar/'
path_input = path_infolder + 'input/'
path_output = path_infolder + f'output/output{scenario}/'

print('Files will be saved at:\n', path_output)
os.makedirs(path_output, exist_ok=True)


path_root_intensity = 'Users/desgranc/Documents/work/projects/HD120326/data/reduced/IRDAP/2018-06-01/reduced_pdi/star_pol_subtr/HIP67497_2018-06-02_Q_phi_star_pol_subtr.fits'

# Synthetic disk    
if scenario == '_HD120326' :
    offset = 0 
    fake_disk_map = fits.getdata('/Users/desgranc/Documents/work/GitHub_clone/projects/HD102326/fm/fm_test_polar/input/polar_amoeba_final/disk_model.fits')


elif scenario == '_Engler2018' : # Example Engler+2018
    offset = np.pi*22/180
    model = ScatteredLightDisk(nx=100,
                               ny=100,
                               distance=50,
                               itilt=104,
                               omega=0,
                               pxInArcsec=12.25e-3,
                               pa= -90, #110,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': 10,
                                   'aout': -5,
                                   'a': 10,
                                   'e': 0,
                                   'ksi0':  0.1756,
                                   'gamma': 2.,
                                   'beta': 0.45,
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': 0.6,
                                   'polar': True
                               })
    
    fake_disk_map = model.compute_scattered_light()


elif scenario == '_HeikampKeller2019_face_on_close' : # Example HeikampKeller2019_face_on_close
    offset = 0 #np.pi*22/180
    model = ScatteredLightDisk(nx=100,
                               ny=100,
                               distance=50,
                               itilt=0,
                               omega=0,
                               pxInArcsec=12.25e-3,
                               pa= 0, #110,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': 2,
                                   'aout': -2,
                                   'a': 5,
                                   'e': 0,
                                   'ksi0':  0.1756,
                                   'gamma': 2.,
                                   'beta': 0.45,
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': 0.6,
                                   'polar': True
                               })
    
    fake_disk_map = model.compute_scattered_light()


elif scenario == '_HeikampKeller2019_face_on_far' : # Example HeikampKeller2019_face_on_close
    offset = 0 #np.pi*22/180
    model = ScatteredLightDisk(nx=300,
                               ny=300,
                               distance=50,
                               itilt=0,
                               omega=0,
                               pxInArcsec=12.25e-3,
                               pa= 0, #110,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': 10,
                                   'aout': -5,
                                   'a': 50,
                                   'e': 0,
                                   'ksi0':  0.1756,
                                   'gamma': 2.,
                                   'beta': 0.45,
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': 0.6,
                                   'polar': True
                               })
    
    fake_disk_map = model.compute_scattered_light()



elif scenario == '_HeikampKeller2019_edge_on_close' : # Example HeikampKeller2019_face_on_close
    offset = 0 #np.pi*22/180
    model = ScatteredLightDisk(nx=100,
                               ny=100,
                               distance=50,
                               itilt=0,
                               omega=0,
                               pxInArcsec=12.25e-3,
                               pa= 0, #110,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': 2,
                                   'aout': -2,
                                   'a': 5,
                                   'e': 0,
                                   'ksi0':  0.1756,
                                   'gamma': 2.,
                                   'beta': 0.45,
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': 0.6,
                                   'polar': True
                               })
    
    fake_disk_map = model.compute_scattered_light()

    fake_disk_map[50::,:]=0


elif scenario == '_HeikampKeller2019_edge_on_far' : # Example HeikampKeller2019_face_on_close
    offset = 0 #np.pi*22/180
    model = ScatteredLightDisk(nx=300,
                               ny=300,
                               distance=50,
                               itilt=0,
                               omega=0,
                               pxInArcsec=12.25e-3,
                               pa= 0, #110,
                               density_dico={
                                   'name': '2PowerLaws',
                                   'ain': 10,
                                   'aout': -5,
                                   'a': 50,
                                   'e': 0,
                                   'ksi0':  0.1756,
                                   'gamma': 2.,
                                   'beta': 0.45,
                               },
                               spf_dico={
                                   'name': 'HG',
                                   'g': 0.6,
                                   'polar': True
                               })
    
    fake_disk_map = model.compute_scattered_light()
    fake_disk_map[150::,:]=0





print('Sum fake disk', np.nansum(fake_disk_map ))
print('Max fake disk', np.nanmax(fake_disk_map ))
fake_disk_map = fake_disk_map/np.nanmax(fake_disk_map)

print('Sum fake disk', np.nansum(fake_disk_map ))
print('Max fake disk', np.nanmax(fake_disk_map ))
fake_disk_map = fake_disk_map/np.nanmax(fake_disk_map)


# PSF
psf = fits.getdata('/Users/desgranc/Documents/work/GitHub_clone/projects/HD102326/fm/fm_test_polar/input/polar_inputs/PSF.fits')
print('Sum PSF', np.nansum(psf))
#psf_irdis_cropped = psf[512-16:512+16,512-16:512+16] 
psf_irdis_cropped = psf  #psf_irdis_cropped/np.sum(psf_irdis_cropped)
#ds9.display(psf_irdis_cropped)

#angarr = angle_array(fake_disk_map.shape)
angarr =  compute_im_pa_grid(fake_disk_map, return_unit='rad')  + offset

if save_files: fits.writeto(path_output+'angarr.fits', angarr,overwrite=True)

#
print('=== Compute Stokes vectors ===')

#ds9.display(fake_disk_map)

Q = -fake_disk_map*np.cos(2*angarr)
U = -fake_disk_map*np.sin(2*angarr)

#ds9.display(Q,U)
if save_files: 
    fits.writeto(path_output+'fake_disk_map.fits', fake_disk_map, overwrite=True)
    fits.writeto(path_output+'Q.fits', Q, overwrite=True)
    fits.writeto(path_output+'U.fits', U, overwrite=True)
    
fake_disk_convoled = signal.fftconvolve(fake_disk_map,psf_irdis_cropped, mode='same')

print('Convolve the synthetic image')

if save_files: 
    fits.writeto(path_output+'fake_disk_convolved_map.fits', fake_disk_convoled, overwrite=True)

print('Compute tmp Q, U')

fake_disk_map_tmp_Q=fake_disk_map*np.cos(2*angarr) # the only difference with Q is the sign '-'
fake_disk_map_tmp_U=fake_disk_map*np.sin(2*angarr) # the only difference with U is the sign '-'


print('Convolve tmp Q, U')
fake_disk_map_tmp_Q_convolved = signal.fftconvolve(fake_disk_map_tmp_Q,psf_irdis_cropped, mode='same')
fake_disk_map_tmp_U_convolved = signal.fftconvolve(fake_disk_map_tmp_U,psf_irdis_cropped, mode='same')

print('Compute the rigorous convolve disk image')

fake_disk_map_convolved_rigourous_Qphi = (fake_disk_map_tmp_Q_convolved*np.cos(2*angarr)+fake_disk_map_tmp_U_convolved*np.sin(2*angarr)) # corresponds to Qphi 
fake_disk_map_convolved_rigourous_Uphi = (fake_disk_map_tmp_Q_convolved*np.sin(2*angarr)-fake_disk_map_tmp_U_convolved*np.cos(2*angarr)) # corresponds to Uphi 
fake_disk_map_tmp_pi_convolved = np.sqrt(fake_disk_map_tmp_Q_convolved**2+fake_disk_map_tmp_U_convolved**2)

#ds9.display(fake_disk_map_tmp_Q_convolved,fake_disk_map_tmp_U_convolved,fake_disk_map_convolved_rigourous,fake_disk_convoled)

if save_files: 
    fits.writeto(path_output+'minus_Q.fits', fake_disk_map_tmp_Q, overwrite=True)
    fits.writeto(path_output+'minus_U.fits', fake_disk_map_tmp_U, overwrite=True)
    fits.writeto(path_output+'minus_Q_convolved.fits', fake_disk_map_tmp_Q_convolved, overwrite=True)
    fits.writeto(path_output+'minus_U_convolved.fits', fake_disk_map_tmp_U_convolved, overwrite=True)
    fits.writeto(path_output+'Qphi_convolved_rigourous.fits', fake_disk_map_convolved_rigourous_Qphi, overwrite=True)
    fits.writeto(path_output+'Uphi_convolved_rigourous.fits', fake_disk_map_convolved_rigourous_Uphi, overwrite=True)
    fits.writeto(path_output+'pI_convolved_rigourous.fits', fake_disk_map_tmp_pi_convolved, overwrite=True)


#ds9.display(fake_disk_map_convolved_rigourous,fake_disk_convoled)
    
print('=== Compute the residuals ===')

residuals = fake_disk_convoled-fake_disk_map_convolved_rigourous_Qphi 

relative_change = residuals/fake_disk_map_convolved_rigourous_Qphi 

#ds9.display(fake_disk_map_convolved_rigourous,relative_change)
if save_files: 
    fits.writeto(path_output+'residuals.fits', residuals, overwrite=True)
    fits.writeto(path_output+'relative_change.fits', relative_change, overwrite=True)


print('=== Mask the pixels if the relative change is inferior to 0.01 ===')

id_fainter_1percent = fake_disk_map_convolved_rigourous_Qphi <0.01

relative_change_masked = np.copy(relative_change)
relative_change_masked[id_fainter_1percent] = np.nan

#ds9.display(fake_disk_map_convolved_rigourous,relative_change_masked)

if save_files: 
    fits.writeto(path_output+'relative_change_masked.fits', relative_change_masked, overwrite=True)

#np.nanmax(np.abs(relative_change_masked))


#ds9.display(fake_disk_convoled,fake_disk_map_tmp_pi_convolved,residuals)
# this shows that there is no change whether we derive Q and U and pI and then convolve
#print(np.max(residuals)/np.max(fake_disk_convoled)) #0.0015
# ds9.display(Q_phi_frame_cropped,fake_disk_convoled)
