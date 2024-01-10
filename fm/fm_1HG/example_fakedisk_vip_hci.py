#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:22:02 2019

@author: jmilli
"""

import vip_hci as vip
ds9=vip.Ds9Window()


pixel_scale=0.01225 # pixel scale in arcsec/px
dstar= 80 # distance to the star in pc
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

# Warning: 
#        The star is assumed to be centered at the frame center as defined in
#        the vip_hci.var.frame_center function, e.g. at pixel [ny//2,nx//2]
#        (if odd: nx/2-0.5 e.g. the middle of the central pixel 
#        if even: nx/2)

itilt = 76 # inclination of your disk in degreess
pa= 30 # position angle of the disk in degrees (from north to east)
a = 70 # semimajoraxis of the disk in au 


# 1. Let's try a pole-on symmetric diks without anisotropy of scattering
fake_disk1 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=0,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-12,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'HG', 'g':0., 'polar':False},flux_max=1)
fake_disk1_map = fake_disk1.compute_scattered_light()
ds9.display(fake_disk1_map)


# 2. Let's incline it by 76º (itilt)
fake_disk2 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-12,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'HG', 'g':0., 'polar':False},flux_max=1)
fake_disk2_map = fake_disk2.compute_scattered_light()

ds9.display(fake_disk2_map)


# 3. Let's change the scattering phase function and add some anisotropy of scattering
fake_disk3 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-12,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'HG', 'g':0.4, 'polar':False},flux_max=1)
fake_disk3_map = fake_disk3.compute_scattered_light()
ds9.display(fake_disk3_map)

# 4. Let's change the scattering phase function and add some backward scattering
fake_disk4 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-12,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'DoubleHG','g':[0.6,-0.6],'weight':0.7,\
                                  'polar':False},flux_max=1)
fake_disk4_map = fake_disk4.compute_scattered_light()
ds9.display(fake_disk4_map)

# 5. Let's plug in the Rayleigh scattering in polarimetry
fake_disk5 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':12,'aout':-12,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'DoubleHG','g':[0.6,-0.6],'weight':0.7,\
                                  'polar':True},flux_max=1)
fake_disk5_map = fake_disk5.compute_scattered_light()
ds9.display(fake_disk5_map)


#%% Let's model another disk similar to HD114082

pixel_scale=0.01225 # pixel scale in arcsec/px

nx = 121 # number of pixels of your image in X
ny = 121 # number of pixels of your image in Y

PA = -90+15.
itilt=80.
e=0.0
omega=0.
aniso_g=0
gamma=2.
ksi0=1.
dstar=85.
r0=24.*pixel_scale*dstar
ain=10.
aout=-4.

r0_HD114082 = 30.08429406
PA_HD114082 = -73.71558094
itilt_HD114082 = 81.36660035

alphaout_HD114082 = -4.9849274
# flux_HD114082 = 965.16694895


fake_disk_HD114082 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt_HD114082,omega=omega,pxInArcsec=pixel_scale,pa=PA_HD114082,\
                        density_dico={'name':'2PowerLaws','ain':ain,'aout':alphaout_HD114082,\
                        'a':r0_HD114082,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                        spf_dico={'name':'HG', 'g':aniso_g, 'polar':False})

fake_disk_HD114082.print_info()


# -----------------------------------
# Geometrical properties of the image
# -----------------------------------
# Image size: 121 px by 121 px
# Pixel size: 0.0123 arcsec/px or 1.04 au/px
# Distance of the star 85.0 pc
# From -62.5 au to 62.5 au in X
# From -62.5 au to 62.5 au in Y
# Position angle of the disc: -73.72 degrees
# Inclination 81.37 degrees
# Argument of pericenter 0.00 degrees
# ----------------------------
# Dust distribution parameters
# ----------------------------
# Reference semi-major axis: 30.1au
# Semi-major axis at maximum dust density in plane z=0: 30.8au (same as ref sma if ain=-aout)
# Semi-major axis at half max dust density: 27.3au / 36.8au for the inner/outer edge, or a FWHM of 9.6au
# Ellipse p parameter: 30.1au
# Ellipticity: 0.000
# Inner slope: 10.00
# Outer slope: -4.98
# Density at the reference semi-major axis: 1.000e+00 (arbitrary unit
# Scale height: 1.00 au at 30.08
# Vertical profile index: 2.00
# Disc vertical FWHM: 1.10 at 30.08
# Flaring coefficient: 1.00
# ------------------------------------
# Properties for numerical integration
# ------------------------------------
# Requested accuracy 5.00e-03
# Maximum radius for integration: 87.08 au
# Maximum height for integration: 2.30 au
# Inclination threshold: 88.49 degrees
# ----------------------------
# Phase function parameters
# ----------------------------
# Type of phase function: HG
# Linear polarisation: False
# Heynyey Greenstein coefficient: 0.00

fake_disk_HD114082_map = fake_disk_HD114082.compute_scattered_light()


ds9.display(fake_disk_HD114082_map)


#%%
import numpy as np

# we create a synthetic cube of 30 images
nframes = 30
# we assume we have 30º of parallactic angle rotation centered around meridian
derotation_angles = np.arange(-15,15,1)

# we create a cube with the disk injected at the correct parallactic angles:
cube_fake_disk1 = vip.metrics.cube_inject_fakedisk(fake_disk1_map,derotation_angles)
cube_fake_disk2 = vip.metrics.cube_inject_fakedisk(fake_disk2_map,-derotation_angles)

cadi_fakedisk2 = vip.medsub.median_sub(cube_fake_disk2,derotation_angles)
ds9.display(fake_disk2_map,cadi_fakedisk2)

# Now if have a PSF and want to also convolve the disk with the PSF, we can do that ! 
# Let's first create a syntehtic gaussian PSF
def gaus2d(x=0, y=0, mx=0, my=0, sx=2, sy=2):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
x = np.linspace(-5, 5,num=11)
y = np.linspace(-5, 5,num=11)
x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
psf = gaus2d(x, y)
ds9.display(psf)

# Then we inject the disk in the cube and convolve by the PSF
cube_fake_disk2_convolved = vip.metrics.cube_inject_fakedisk(fake_disk2_map,-derotation_angles,psf=psf)

cadi_fakedisk2_convolved = vip.medsub.median_sub(cube_fake_disk2_convolved,derotation_angles)
ds9.display(fake_disk2_map,cadi_fakedisk2,cadi_fakedisk2_convolved)




