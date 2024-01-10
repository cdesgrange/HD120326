"""
NAME:
 import_initialization_dataframes

PURPOSE:
 Import packages usually used

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 21/10/01, Written by Célia DESGRANGE, IPAG/MPIA
 21/12/22, Modified by Célia DESGRANGE, commented the path for astrometry table (Kervella)
 22/01/06, Modified by Célia DESGRANGE, added paths:
    .specalcharac: (line 26) folder_all_specal_charac, (line 34) list_obj_specalcharac
    .variables: (line 38+) dF_prop_name_cols,NAME_OBJ, PARALLAX(_ERR)
22/01/13, Modified by Célia DESGRANGE, added variables:
    .MAG_STAR_J, MAG_STAR_H, MAG_STAR_K (line 48)
"""

from import_packages_generic import *

infolder_super_earths = '/Users/user/Documents/desgrange/super-earths/'
folder_download_and_diplay = 'download_and_display_scripts/'
folder_data_scripts   = 'download_and_display_scripts/download_scripts/'
folder_sphere_DC_data = 'download_and_display_scripts/SPHERE_DC_DATA/'
folder_all_specal_charac = '*charac*/*/*/'
folder_MESS2 = 'softwares_and_documentation/MESS2/'
subfolder_model = 'contrast_to_mass/models/'
subfolder_detlim_mass_map = 'contrast_to_mass/input/reduced_images/'
list_obj_obs  = 'csv_tables/manual_statistics_for_wiki_working_document_v3.csv'
list_obj_prop = 'csv_tables/names.csv'
list_obj_specalcharac = 'comp_astro_photo_tdb.csv'
list_planets = 'csv_tables/planets.txt'
model_cond = 'model_cond2003_irdis.csv'
#
dF_obs  = pd.read_csv(infolder_super_earths+list_obj_obs,sep=';')
dF_prop = pd.read_csv(infolder_super_earths+list_obj_prop,sep=';')
dF_prop_name_cols = dF_prop.columns
dF_planets = pd.read_csv(infolder_super_earths+list_planets,sep=';')
NIGHT_OBS = dF_obs['night_obs']
NAME_OBJ  = np.array(dF_prop['target name'])
PARALLAX  = np.array(dF_prop['parallax']) ; PARALLAX_ERR  = np.array(dF_prop['err parallax'])
DISTANCE  = dF_prop['distance']
MAG_STAR_J, MAG_STAR_H, MAG_STAR_K =  np.array(dF_prop['mag star J']), np.array(dF_prop['mag star H']), np.array(dF_prop['mag star K'])
# astrometry
#path = Path('/Users/user/Documents/desgrange/Jupyter/projects/super-earths/plot_astrometry')
#table = Table.read(path.joinpath('results/results_Kervella.ecsv'), format='ascii.ecsv')
