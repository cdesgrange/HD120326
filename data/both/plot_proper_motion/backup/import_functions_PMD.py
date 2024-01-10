"""
NAME:
 import_functions_PMD

PURPOSE:
 Import functions used to compute proper motion and plot them

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 2021/10/01, Written by Célia DESGRANGE, IPAG/MPIA
 
CONTRIBUTION:
    Pierre Kervella: functions to extract Gaia/Hipparcos detection limits
"""

from import_packages_generic import *
from import_functions_generic import *
from import_initialization_variables_PMD import *

## Functions used to make nice plots ##

def closer_positions(ra1,dec1,ra2,dec2,dra=0,ddec=0, tolerance=10, unit='pixel'):
    '''
    Find candidates observed in two different epochs.
        Inputs:
            .'ra1' (type=list/array): list/array of the right ascensions for the detections in the 1st epoch
            .'dec1' (type=list/array): list/array of the declinations for the detections in the 1st epoch
            .'ra2' (type=list/array): list/array of the right ascensions for the detections in the 2nd epoch
            .'dec2' (type=list/array): list/array of the declinations for the detections in the 2nd epoch
            (optional, below:)
            .'dra' (type=int/float): shift to apply to the second epoch (proper motion in RA)
            .'ddec' (type=int/float): shift to apply to the second epoch (proper motion in DEC)
            .'tolerance' (type=int/float): difference tolerated. In practice it is the some of the differences between ra1/ra2_closer and dec1/dec2_closer
            .'unit' (type='string'): indicate which unit is considered, but for now, no importance.
        
        Outputs:
            Return the detections observed in two epoch, more precisely:
                .their indexes in 1st epoch
                .their indexes in 2nd epoch
                .their RA and DEC in 2nd epoch (could be removed I guess)
    '''
    ra2_shift = np.array(ra2)-dra
    dec2_shift = np.array(dec2)-ddec
    N_possibilities = np.nanmax([len(ra1),len(ra2)])
    INDEX1, INDEX_CLOSER, RA_CLOSER, DEC_CLOSER = [], [], [], []

    for k in range(N_possibilities):
        wanted = [ra1[k],dec1[k]]
        L      = [ra2_shift,dec2_shift]
        
        index, diff = index_closer(wanted, L, constraints='several', output_diff=True)
        
        # if the closest value is still too far, do not consider it
        if diff[index] < tolerance :
            print(diff[index],wanted,ra2[index],dec2[index])
            INDEX1.append(k)
            INDEX_CLOSER.append(index)
            RA_CLOSER.append(ra2[index])
            DEC_CLOSER.append(dec2[index])
    return np.array(INDEX1),np.array(INDEX_CLOSER), np.array(RA_CLOSER), np.array(DEC_CLOSER)



#
def timedelta_between_2epochs(target,epochs,unit='year'):
    '''
    Return the proper motion expected for background contaminants between two epochs
        Inputs:
            'target' (type=string):
                    example 'HD154088'
            'epochs' (type=list/array of strings):
                    example np.array(['2017-06-21', '2019-07-22', '2021-07-04'])
            (optional)
            'unit' (type=string): precise in which units to return the timedelta
                    Default is in 'year'. Other option implemented: 'day', 'hour',
                    'minute', and 'second'.
    '''
    T = Time(epochs, format='isot', scale='utc')
    timedelta = ((T[1]-T[0])).value
    if unit == 'year': return timedelta/365.25
    elif unit == 'day' : return timedelta
    elif unit == 'hour': return timedelta*24*3600
    elif unit == 'minute': return timedelta*24*60
    elif unit == 'second': return timedelta*24*3600
    else : return print("this unit is not yet implemented")
    

def find_proper_motion_between_2epochs(target,epochs,unit='mas',pixarc=12.25):
 '''
 Return the proper motion expected for background contaminants between two epochs
     Inputs:
         'target' (type=string):
                 example 'HD154088'
         'epochs' (type=list/array of strings):
                 example np.array(['2017-06-21', '2019-07-22', '2021-07-04'])
         (optional below)
         'unit' (type=string): precise in which units to return the proper motion
                 Default is in 'mas'. Other option implemented: 'pixel'.
         'pixarc' (type=float): in case the 'unit' is to 'pixel', precise what is
                 the factor of conversion. Default is 12.25 mas, corresponding to
                 the platescale from the SPHERE-IRDIS instrument.
 '''
 timedelta = timedelta_between_2epochs(target,epochs,unit='year')

 result_table_names = customSimbad.query_object(target)
 pm_ra, pm_dec = result_table_names['PMRA'], -result_table_names['PMDEC']
 pm_ra, pm_dec = float(pm_ra.value)*timedelta, float(pm_dec.value)*timedelta
 if unit == 'mas':
     return pm_ra, pm_dec
 if unit == 'pixel':
     return pm_ra/pixarc, pm_dec/pixarc
 #print("Proper motion per year:", pm_ra, pm_dec)
 #print("Proper motion between the two epochs", pm_ra*timedelta, pm_dec*timedelta)
 return
 

def find_proper_motion_per_year(target,unit='mas',pixarc=12.25):
'''
Return the proper motion expected for background contaminants between two epochs
    Inputs:
        'target' (type=string):
                example 'HD154088'
        (optional below)
        'unit' (type=string): precise in which units to return the proper motion
                Default is in 'mas'. Other option implemented: 'pixel'.
        'pixarc' (type=float): in case the 'unit' is to 'pixel', precise what is
                the factor of conversion. Default is 12.25 mas, corresponding to
                the platescale from the SPHERE-IRDIS instrument.
'''
epochs = ["2021-01-01","2022-01-01"]
return find_proper_motion_between_2epochs(target,epochs,unit=unit,pixarc=pixarc)
