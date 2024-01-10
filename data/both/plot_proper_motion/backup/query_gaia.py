#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:51:29 2020

@author: jmilli
"""

import pandas as pd
import numpy as np
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

dico = {}
for key in keys:
    dico[key]= []

pd_info_stars = pd.DataFrame(dico)
pd_targets= pd.read_csv(path_csv_tables.joinpath('names.csv'))

pd_info_stars['target name'] = pd_targets['target name']
pd_info_stars['distance (pc)'] = 1/(pd_info_stars['PLX_VALUE']/1000)
pd_info_stars['distance error (pc)'] = pd_info_stars['distance (pc)']**2*(pd_info_stars['PLX_ERROR']/1000)

nb_stars_irdis_fov = np.zeros(len(pd_targets),dtype=int)
nb_stars_ifs_fov = np.zeros(len(pd_targets),dtype=int)
mag_limit_irdis = np.zeros(len(pd_targets),dtype=int)
mag_limit_ifs = np.zeros(len(pd_targets),dtype=int)
for i in range(len(pd_info_stars)):
    long_gal_str =  pd_info_stars['RA_gal'][i]
    lat_gal_str = pd_info_stars['DEC_gal'][i]
    coord_gal = SkyCoord(long_gal_str,lat_gal_str,frame='galactic',unit=(u.deg, u.deg))
    long_gal_float = float(coord_gal.l/(1.*u.deg))
    lat_gal_float = float(coord_gal.b/(1.*u.deg))
    Kmag_limit_ifs = pd_info_stars['FLUX_K'][i]+2.5*5
    Kmag_limit_irdis = pd_info_stars['FLUX_K'][i]+2.5*6
    if Kmag_limit_irdis>21:
        Kmag_limit_irdis = 20.
    if Kmag_limit_ifs>21:
        Kmag_limit_ifs = 20.
    mag_limit_irdis[i] =  Kmag_limit_irdis
    mag_limit_ifs[i] =   Kmag_limit_ifs
    if add_besancon_info :
        nb_stars_irdis_fov[i] = besancon.get_star_count_below_Kmag(long_gal_float,lat_gal_float,Kmag_limit_irdis,fov_irdis_arcsec2)
        nb_stars_ifs_fov[i] = besancon.get_star_count_below_Kmag(long_gal_float,lat_gal_float,Kmag_limit_ifs,fov_ifs_arcsec2)

pd_info_stars['star counts in IRDIS FoV'] = nb_stars_irdis_fov
pd_info_stars['star counts in IFS FoV']   = nb_stars_ifs_fov
if add_besancon_info :
    pd_info_stars['mag limit K IRDIS for Besancon'] = mag_limit_irdis
    pd_info_stars['mag limit K IFS for Besancon']   = mag_limit_ifs

pd_info_stars.to_csv(path_csv_tables.joinpath('target_properties.csv'))


#%%
# target_name = 'GJ674'
# id_target_name = 12
# date = dt.datetime(2017,7,14,2,10)
# date_obs = Time(date)
# print(date_obs.jyear_str)

pd_statistics = pd.read_csv(path_csv_tables.joinpath('statistics.csv'))
nb_contaminants_from_Gaia = np.zeros(len(pd_statistics),dtype=int)
if add_besancon_info :
    mag_lim_K_besancon = np.zeros(len(pd_statistics),dtype=int)
    nb_contaminants_from_Besancon = np.zeros(len(pd_statistics),dtype=int)

for i_stat,date_obs_str in enumerate(pd_statistics['date start']):
#for i_stat in range (37,38):
    date_obs_str = pd_statistics['date start'][i_stat]
    target_name = pd_statistics['target name'][i_stat]
    if target_name != 'GJ674': continue
    date_obs = Time(date_obs_str)

    id_info_star = np.where(pd_info_stars['target name']==target_name)[0][0]

    if target_name == 'BD061339':
        target_name = 'BD-06 1339'
    camera = pd_statistics['camera'][i_stat]

    if add_besancon_info :
        # Now we need to find the ID of the entry in pd_info_stars to populate pd_statistics
        # with the expected contaminants from the Besancon model
        if camera=='IRDIS':
            mag_lim_K_besancon[i_stat] = pd_info_stars['mag limit K IRDIS for Besancon'][id_info_star]
            nb_contaminants_from_Besancon[i_stat] = pd_info_stars['star counts in IRDIS FoV'][id_info_star]
        elif camera=='IFS':
            mag_lim_K_besancon[i_stat] = pd_info_stars['mag limit K IFS for Besancon'][id_info_star]
            nb_contaminants_from_Besancon[i_stat] = pd_info_stars['star counts in IFS FoV'][id_info_star]
        else:
            print('Problem')


    # result_table = Simbad.query_object(target_name)

    result_table_names = Simbad.query_objectids(target_name)

    id_gaia_dr2 = np.nan
    for name in result_table_names:
        if name[0].startswith('Gaia DR2'):
            id_gaia_dr2 = int(name[0][name[0].index('Gaia DR2')+8:])
            print('\nFrom Gaia for target',target_name,'=',name[0])
    if not np.isfinite(id_gaia_dr2):
        print('Gaia DR2 ID not found')
        continue

    # # Unnecessary script, just to test
    # query = "SELECT gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec,gaia_source.radial_velocity FROM gaiadr2.gaia_source WHERE source_id = '{0:d}'".format(id_gaia_dr2)

    # 1. here we propagate the coordinate  of the source to the epoch of observation
    query1 = """SELECT
            EPOCH_PROP(gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec,gaia_source.radial_velocity,2015.5,{0:.6f})
            FROM gaiadr2.gaia_source
            WHERE source_id = '{1:d}'""".format(date_obs.jyear,id_gaia_dr2)

    j1 = Gaia.launch_job(query1)
    r1 = j1.get_results()

    r1_data = r1[0][0].data
    print('\nOur source: \nRA={}\nDEC={} \nPLX={} \nPM RA={} \nPM DEC={} \nPM RADIAL={}'.format(r1_data[0],r1_data[1],r1_data[2],r1_data[3],r1_data[4],r1_data[5]))
    # epoch prop[1]: ra, Right Ascension (deg)
    # epoch prop[2]: dec, Declination (deg)
    # epoch prop[3]: plx, Parallax (mas)
    # epoch prop[4]: pmra, Proper Motion in Right Ascension (mas/yr)
    # epoch prop[5]: pmde, Proper Motion in Declination (mas/yr)
    # epoch prop[6]: pmr, Radial Proper Motion (mas/yr)

    if camera =='IRDIS':
        search_radius_arcsec = 6.25 # we can increase this value for tests...
    elif camera == 'IFS':
        search_radius_arcsec = 0.8

    search_radius_degree = search_radius_arcsec/3600.

    # 2. Then we query all GAIA DR2 sources within 10 arcsec from the source at the epoch of observations
    #    (assuming the sources do not have signifcant proper motion)
    query2 = """SELECT DISTANCE(
      POINT('ICRS', ra, dec),
      POINT('ICRS', {0:f}, {1:f})) AS dist, *
    FROM gaiadr2.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {0:f}, {1:f}, {2:f}))
    ORDER BY dist ASC""".format(r1['epoch_prop'][0][0],r1['epoch_prop'][0][1],search_radius_degree)

    j2 = Gaia.launch_job(query2)
    r2 = j2.get_results()
    nstars2 = len(r2)
    print('\nAll sources:\n',r2)

    # 3. For each source found, we propagate the proper motion from 2015.5 to the epoch of observation
    #    and derive the distance between the target and the contaminant.

    dico_contaminants = {\
        'ID Gaia DR2':[],'RA (deg)':[],'DEC (deg)':[],'pmra':[],'pmdec':[],\
        'teff_val':[],'delta RA (arcsec)':[],'delta DEC (arcsec)':[],\
        'separation (arcsec)':[],'PA (deg)':[],'parallax':[],\
        'phot_g_mean_mag':[],'phot_bp_mean_mag':[],'phot_rp_mean_mag':[],\
        'FLUX_J':[],'FLUX_H':[],'FLUX_K':[]\
        }

    for i,id_source in enumerate(r2['source_id']):
        if id_source == id_gaia_dr2:
            print('The {0:d}th source is actualy the target itself. '.format(id_source))
        else:
            query3 = """SELECT EPOCH_PROP(
                gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec,gaia_source.radial_velocity,2015.5,{0:.6f})
            FROM gaiadr2.gaia_source
            WHERE source_id = '{1:d}'""".format(date_obs.jyear,id_source)
            j3 = Gaia.launch_job(query3)
            r3 = j3.get_results()
            # print(r3)
            dico_contaminants['ID Gaia DR2'].append(id_source)
            delta_ra_deg = (r3['epoch_prop'][0][0]-r1['epoch_prop'][0][0])*np.cos(np.deg2rad(r1['epoch_prop'][0][1]))
            delta_dec_deg = r3['epoch_prop'][0][1]-r1['epoch_prop'][0][1]
            delta_ra_arcsec = delta_ra_deg*3600
            delta_dec_arcsec = delta_dec_deg*3600
            dico_contaminants['RA (deg)'].append(r3['epoch_prop'][0][0])
            dico_contaminants['DEC (deg)'].append(r3['epoch_prop'][0][1])
            dico_contaminants['delta RA (arcsec)'].append(delta_ra_arcsec)
            dico_contaminants['delta DEC (arcsec)'].append(delta_dec_arcsec)
            for key in ['pmra','pmdec','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','teff_val','parallax']:
                if np.ma.is_masked(r2[key][i]):
                    dico_contaminants[key].append(np.nan)
                else:
                    dico_contaminants[key].append(r2[key][i])
            distance = np.sqrt(delta_ra_arcsec**2+delta_dec_arcsec**2)
            pa = np.rad2deg(np.arctan2(delta_ra_arcsec,delta_dec_arcsec))
            dico_contaminants['separation (arcsec)'].append(distance)
            dico_contaminants['PA (deg)'].append(pa)

            print('Companion found at delta RA={0:.3f}arcsec and delta DEC={1:.3f}arcsec'.format(delta_ra_arcsec,delta_dec_arcsec))
            print('Companion found at {0:.3f}arcsec PA {1:.2f}'.format(distance,pa))
            print('Gaia DR2',id_source)

            result_table_names = customSimbad.query_object('Gaia DR2 {0:d}'.format(id_source))
            try:
                for band in ['FLUX_J','FLUX_H','FLUX_K']:
                    if not np.ma.is_masked(result_table_names[band][0]):
                        dico_contaminants[band].append(result_table_names[band][0])
                    else:
                        dico_contaminants[band].append(np.nan)
            except TypeError:
                for band in ['FLUX_J','FLUX_H','FLUX_K']:
                        dico_contaminants[band].append(np.nan)
    pd_contaminants = pd.DataFrame(dico_contaminants)
    pd_contaminants['distance'] = 1/(pd_contaminants['parallax']/1000.)
    nb_contaminants_from_Gaia[i_stat] = len(pd_contaminants)

    if nb_contaminants_from_Gaia[i_stat]>0:
        pd_contaminants.to_csv(path_gaia.joinpath('contaminants_{0:s}_{1:s}_{2:s}.csv'.format(target_name,date_obs.datetime.date().isoformat(),camera)))
        # then we create a ds9 arrow pointing at the source.
        reg_string = \
    """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image"""
        for j,sep in enumerate(pd_contaminants['separation (arcsec)']):
            pa_ds9 = pd_contaminants['PA (deg)'][j]+90
            sep_irdis = sep/0.01225
            reg_string = reg_string+"""
    # vector(725,725,{0:.2f},{1:.2f}) vector=1""".format(sep_irdis,pa_ds9)
        filename = path_gaia.joinpath('contaminants_{0:s}_{1:s}_{2:s}_for_image_1448x1448.reg'.format(target_name,date_obs.datetime.date().isoformat(),camera))
        txtfile = open(filename,'w')
        txtfile.write(reg_string)
        txtfile.close()

pd_statistics['nb contaminants from Gaia'] = nb_contaminants_from_Gaia
if add_besancon_info :
    pd_statistics['K band limit mag for Besancon'] = mag_lim_K_besancon
    pd_statistics['nb contaminants from Besancon'] = nb_contaminants_from_Besancon
# pd_statistics.to_csv(path_csv_tables.joinpath('statistics_with_Gaia_contaminants_test_celia.csv'))

# only 8 known contaminants from Gaia
print('In total {0:d} contaminants are known from Gaia'.format(np.sum(nb_contaminants_from_Gaia)))
# In total 8 contaminants are known from Gaia
