"""
NAME:
 import_packages_PMD

PURPOSE:
 Import specifc packages to plot proper motion diagrams (PMD)

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 2022/01/13, Written by Célia DESGRANGE, IPAG/MPIA
 
"""
from import_packages_generic import *

# specific importations for this Jupyter notebook
import proper_motion
#import query_gaia

# Some astro stuff
import datetime as dt
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy import units as u
import astropy.io.fits as fits
from astropy.constants import c
from astropy.time import Time
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

# Query Simbad -> retrieve RA, DEC, and proper motion from the star
customSimbad = Simbad()
fields = ['flux(V)','flux(G)','flux(R)','flux(J)','flux(H)','flux(K)',\
          'flux_error(V)','flux_error(G)','flux_error(R)','flux_error(J)',\
          'flux_error(H)','flux_error(K)',\
          'sp','pmra','pmdec',\
          'ra(gal)','dec(gal)',\
          'pm_err_angle','pm_err_maja','pm_err_mina','pm_bibcode','pm_qual',\
          'plx','plx_error','plx_bibcode',\
          'rv_value','rvz_error','rvz_qual','rvz_bibcode','rvz_radvel']

customSimbad.add_votable_fields(*fields)

keys = ['RA','DEC',\
        'PMRA','PMDEC','PM_ERR_MAJA','PM_ERR_MINA','PM_ERR_ANGLE','PM_QUAL','PM_BIBCODE',\
        'PLX_VALUE','PLX_ERROR',\
        'RA_gal','DEC_gal']
