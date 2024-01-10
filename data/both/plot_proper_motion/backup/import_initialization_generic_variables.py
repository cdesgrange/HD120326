"""
NAME:
 import_initialization_generic_variables

PURPOSE:
 Define colors,  conversion factors

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 2021/10/01, Written by Célia DESGRANGE, IPAG/MPIA
 2021/11/25, Modified by Célia DESGRANGE, IPAG/MPIA
 
"""
from import_packages_generic import *

## Colors ##
color_orange = [1,0.3,0]
color_gray   = [0.2,0.2,0.2]
color_green  = [0,0.35,0.2]
color_purple = [0.3,0,0.6]
color_red    = [0.9,0,0]
color_blue   = [0,0,1]

list_colors_orange  = ['white','tomato',[0.6,0,0]]
list_colors_rainbow = ['red','orange','gold','green','blue']

## Conversion factors ##
MEarth = cst.M_earth.value # m
MJup   = cst.M_jup.value   # m
MSun   = cst.M_sun.value   # m

MSun2MJup = MSun/MEarth
MJup2MEarth = MJup/MEarth
MSun2MEarth = MSun/MEarth

REarth = cst.R_earth.value # m
RJup   = cst.R_jup.value   # m
RSun   = cst.R_sun.value   # m

RSun2REarth = RSun/REarth
RJup2REarth = RJup/REarth
RSun2RJup   = RSun/RJup


## Instrument SPHERE ##
instru = "SPHERE"
IWA_size = 0.1 # arcsec
pixarc_ird = 12.25 ; pixarc_ifs = 7.46
