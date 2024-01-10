"""
NAME:
 import_initialization_CaPMD_variables

PURPOSE:
 Initialize all variables required to plot CMD and PMD in the context of the Super-Earth survey

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 2021/10/04, Written by Célia DESGRANGE, IPAG/MPIA
 2022/01/13, Updated by Célia DESGRANGE, IPAG/MPIA with the version from plot_proper_motion notebook
    . extra variables were defined as SEP_H23/K12, PA_H23/K12 idem with ERR
    . SNR in H2/K1 should be > 5 and in H3/K2 should be > 0.1 (avoir a problem with HD154088)
"""

from import_packages_generic import *
from import_initialization_generic_variables import *
from import_initialization_dataframes import *



files = infolder_super_earths + folder_sphere_DC_data + folder_all_specal_charac + list_obj_specalcharac

# Initialization lists for magnitudes(+err): relative, apparent, absolute
MAG_REL_H2, MAG_REL_H3, MAG_REL_K1, MAG_REL_K2 = [], [], [], []
MAG_REL_ERR_H2, MAG_REL_ERR_H3, MAG_REL_ERR_K1, MAG_REL_ERR_K2 = [], [], [], []
MAG_APP_H2_companion, MAG_APP_H3_companion, MAG_APP_K1_companion, MAG_APP_K2_companion = [], [], [], []
MAG_APP_ERR_H2_companion, MAG_APP_ERR_H3_companion, MAG_APP_ERR_K1_companion, MAG_APP_ERR_K2_companion = [], [], [], []
MAG_ABS_H2_companion, MAG_ABS_H3_companion, MAG_ABS_K1_companion, MAG_ABS_K2_companion = [], [], [], []
MAG_ABS_ERR_H2_companion, MAG_ABS_ERR_H3_companion, MAG_ABS_ERR_K1_companion, MAG_ABS_ERR_K2_companion = [], [], [], []

TARGET_MAG_H23, TARGET_MAG_K12 = [], []
SNR_H23, SNR_K12 = [], []
NAME_H23, NAME_K12 = [], []
X_MAS_H23, Y_MAS_H23 = [],[] ; X_MAS_K12, Y_MAS_K12 = [],[]
X_MAS_ERR_H23, Y_MAS_ERR_H23 = [],[] ; X_MAS_ERR_K12, Y_MAS_ERR_K12 = [],[]

SEP_H23, PA_H23 = [],[] ; SEP_K12, PA_K12 = [],[]
SEP_ERR_H23, PA_ERR_H23 = [],[] ; SEP_ERR_K12, PA_ERR_K12 = [],[]

DISTANCE_H23, DISTANCE_K12 = [], []
MAG_STAR_H23, MAG_STAR_K12 = [], []

for name in glob(files):
    print('\n',name)
    dF = pd.read_csv(name,sep=',')
    name_target = dF['target'][0].replace('-','')
    if name_target == 'HD285968':
        name_target = 'GJ176'
    elif name_target == 'GJ667C':
        name_target = 'GJ667'
    elif name_target == 'HD42581':
        name_target = 'GJ229'
    
    #name_target_csv = ' ' +name_target+ ' '
    #print('target', name_target, 'd_star', d_star)
    parallax_star = float(dF_prop[dF_prop['target name']==name_target]['parallax'])*1e-3
    parallax_err_star = float(dF_prop[dF_prop['target name']==name_target]['err parallax'])*1e-3
    d_star = 1/parallax_star
    d_err_star = parallax_err_star/parallax_star**2
    print('target', name_target, 'd_star', d_star, 'd_err_star', d_err_star)
    
    n = len(dF)//2
    print('n',n)
    for i in range(n):
        if dF["filter"][0] == 'DB_H23' and dF["snr"][i] >= 5 and dF["snr"][i+n] > 0.1: #and dF['algorithm'][0] == 'Classical_ADI':
            mag_star_H = float(dF_prop[dF_prop['target name']==name_target]['mag star H'])
            # relative magnitude
            MAG_REL_H2.append(dF["mag"][i]); MAG_REL_H3.append(dF["mag"][i+n])
            MAG_REL_ERR_H2.append(dF["magerr"][i]); MAG_REL_ERR_H3.append(dF["magerr"][i+n])
            # sometimes error overestimated because psf luminosity variates deeply between beginning
            # and end of the observing sequence -> remove the "mag error psf" from the total error budget
            if dF['magerrpsf'][i] > 1 or dF['magerrpsf'][i+n] > 1 :
                MAG_REL_ERR_H2[-1] -= dF['magerrpsf'][i]*0.95
                MAG_REL_ERR_H3[-1] -= dF['magerrpsf'][i+n]*0.95
            # apparent magnitude
            MAG_APP_H2_companion.append(mag_star_H+MAG_REL_H2[-1])
            MAG_APP_H3_companion.append(mag_star_H+MAG_REL_H3[-1])
            # absolute magnitude
            MAG_ABS_H2_companion.append(MAG_APP_H2_companion[-1]-5*np.log10(d_star/10.))
            MAG_ABS_H3_companion.append(MAG_APP_H3_companion[-1]-5*np.log10(d_star/10.))
            # err
            MAG_ABS_ERR_H2_companion.append(np.sqrt(MAG_REL_ERR_H2[-1]**2 + (5*d_err_star/(d_star*np.log(10)))**2))
            MAG_ABS_ERR_H3_companion.append(np.sqrt(MAG_REL_ERR_H3[-1]**2 + (5*d_err_star/(d_star*np.log(10)))**2))
            print('mag abs=','%.1f'%MAG_ABS_H2_companion[-1],
                  '; color=%.1f'%(MAG_ABS_H2_companion[-1]-MAG_ABS_H3_companion[-1]),
                  '; mag err H2=','%.2f'%MAG_ABS_ERR_H2_companion[-1],'err H3=%.2f'%MAG_ABS_ERR_H3_companion[-1],
                  '; magerrpsf H2=', dF['magerrpsf'][i],'magerrpsf H3=', dF['magerrpsf'][i+n],
                  '; snr=',  dF['snr'][i+n])
            # trace the target name
            TARGET_MAG_H23.append(name_target)
            # and its SNR
            SNR_H23.append([dF["snr"][i],dF["snr"][n+i]])
            #print('M_H2',MAG_ABS_H2_companion[-1], 'H2-H3', MAG_ABS_H2_companion[-1]-MAG_ABS_H3_companion[-1])
            # position
            X_MAS_H23.append(dF["xpos"][i]), Y_MAS_H23.append(dF["ypos"][i])
            X_MAS_ERR_H23.append(dF["xposerr"][i]), Y_MAS_ERR_H23.append(dF["yposerr"][i])
            SEP_H23.append(dF["sepa"][i]), PA_H23.append(dF["pa"][i])
            SEP_ERR_H23.append(dF["sepaerr"][i]), PA_ERR_H23.append(dF["paerr"][i])
            # other
            NAME_H23.append(name)
            DISTANCE_H23.append(d_star)
            MAG_STAR_H23.append(mag_star_H)

                
            
        elif dF["filter"][0] == 'DB_K12' and dF["snr"][i] >= 5 and dF["snr"][i+n] > 0.1 :
            mag_star_K = float(dF_prop[dF_prop['target name']==name_target]['mag star K'])
            # relative magnitude
            MAG_REL_K1.append(dF["mag"][i]); MAG_REL_K2.append(dF["mag"][i+n])
            MAG_REL_ERR_K1.append(dF["magerr"][i]); MAG_REL_ERR_K2.append(dF["magerr"][i+n])
            # sometimes error overestimated because psf luminosity variates deeply between beginning
            # and end of the observing sequence -> remove the "mag error psf" from the total error budget
            if dF['magerrpsf'][i] > 1 or dF['magerrpsf'][i+n] > 1 :
                MAG_REL_ERR_K1[-1] -= dF['magerrpsf'][i]*0.95
                MAG_REL_ERR_K2[-1] -= dF['magerrpsf'][i+n]*0.95
            # apparent magnitude
            MAG_APP_K1_companion.append(mag_star_K+MAG_REL_K1[-1])
            MAG_APP_K2_companion.append(mag_star_K+MAG_REL_K2[-1])
            # absolute magnitude
            MAG_ABS_K1_companion.append(MAG_APP_K1_companion[-1]-5*np.log10(d_star/10.))
            MAG_ABS_K2_companion.append(MAG_APP_K2_companion[-1]-5*np.log10(d_star/10.))
            # err
            MAG_ABS_ERR_K1_companion.append(np.sqrt(MAG_REL_ERR_K1[-1]**2 + (5*d_err_star/(d_star*np.log(10)))**2))
            MAG_ABS_ERR_K2_companion.append(np.sqrt(MAG_REL_ERR_K2[-1]**2 + (5*d_err_star/(d_star*np.log(10)))**2))
            # trace the target name
            TARGET_MAG_K12.append(name_target)
            # and its SNR
            SNR_K12.append([dF["snr"][i],dF["snr"][n+i]])
            # position
            #print('M_K1',MAG_ABS_H2_companion[-1], 'K1-K2', MAG_ABS_K1_companion[-1]-MAG_ABS_K2_companion[-1])
            X_MAS_K12.append(dF["xpos"][i]), Y_MAS_K12.append(dF["ypos"][i])
            X_MAS_ERR_K12.append(dF["xposerr"][i]), Y_MAS_ERR_K12.append(dF["yposerr"][i])
            SEP_K12.append(dF["sepa"][i]), PA_K12.append(dF["pa"][i])
            SEP_ERR_K12.append(dF["sepaerr"][i]), PA_ERR_K12.append(dF["paerr"][i])
            # other
            NAME_K12.append(name)
            DISTANCE_K12.append(d_star)
            MAG_STAR_K12.append(mag_star_K)
            print(#'mag rel err K1','%.2f'%MAG_REL_ERR_K1[-1],'K2','%.2f'%MAG_REL_ERR_K2[-1],
                  'mag abs=','%.1f'%MAG_ABS_K1_companion[-1],
                  '; color=%.1f'%(MAG_ABS_K1_companion[-1]-MAG_ABS_K2_companion[-1]),
                  '; mag err K1=','%.2f'%MAG_ABS_ERR_K1_companion[-1],'err K2=%.2f'%MAG_ABS_ERR_K2_companion[-1],
                  '; magerrpsf K1=', dF['magerrpsf'][i],'magerrpsf K2=', dF['magerrpsf'][i+n],
                  '; magerrseq K2=%.2f'% dF['magerrseq'][i+n],'; magerrfit K2=%.2f'% dF['magerrfit'][i+n],
                  '; snr=',  dF['snr'][i+n],
                  ' with',dF['algorithm'][0],
                  ' at xpix','%.1f'%(724-X_MAS_K12[-1]/pixarc_ird-1),'ypix','%.1f'%(724+Y_MAS_K12[-1]/pixarc_ird+1),
                 )
            

MAG_ABS_H2_companion = np.array(MAG_ABS_H2_companion) ; MAG_ABS_H3_companion = np.array(MAG_ABS_H3_companion)
MAG_ABS_K1_companion = np.array(MAG_ABS_K1_companion) ; MAG_ABS_K2_companion = np.array(MAG_ABS_K2_companion)

MAG_APP_ERR_H2_companion = np.array(MAG_APP_ERR_H2_companion) ; MAG_APP_ERR_H3_companion = np.array(MAG_APP_ERR_H3_companion)
MAG_APP_ERR_K1_companion = np.array(MAG_APP_ERR_K1_companion) ; MAG_APP_ERR_K2_companion = np.array(MAG_APP_ERR_K2_companion)

MAG_ABS_ERR_H2_companion = np.array(MAG_ABS_ERR_H2_companion) ; MAG_ABS_ERR_H3_companion = np.array(MAG_ABS_ERR_H3_companion)
MAG_ABS_ERR_K1_companion = np.array(MAG_ABS_ERR_K1_companion) ; MAG_ABS_ERR_K2_companion = np.array(MAG_ABS_ERR_K2_companion)

MAG_REL_ERR_H2, MAG_REL_ERR_H3 = np.array(MAG_REL_ERR_H2), np.array(MAG_REL_ERR_H3)
MAG_REL_ERR_K1, MAG_REL_ERR_K2 = np.array(MAG_REL_ERR_K1), np.array(MAG_REL_ERR_K2)

TARGET_MAG_H23 = np.array(TARGET_MAG_H23) ; TARGET_MAG_K12 = np.array(TARGET_MAG_K12)
SNR_H23, SNR_K12 = np.array(SNR_H23), np.array(SNR_K12)
DISTANCE_H23 = np.array(DISTANCE_H23); DISTANCE_K12 = np.array(DISTANCE_K12)
MAG_STAR_H23 = np.array(MAG_STAR_H23); MAG_STAR_K12 = np.array(MAG_STAR_K12)
color_H23 = MAG_ABS_H2_companion-MAG_ABS_H3_companion ; color_K12 = MAG_ABS_K1_companion-MAG_ABS_K2_companion

X_MAS_H23, Y_MAS_H23 = np.array(X_MAS_H23), np.array(Y_MAS_H23)
X_MAS_K12, Y_MAS_K12 = np.array(X_MAS_K12), np.array(Y_MAS_K12)
X_MAS_ERR_H23, Y_MAS_ERR_H23 = np.array(X_MAS_ERR_H23), np.array(Y_MAS_ERR_H23)
X_MAS_ERR_K12, Y_MAS_ERR_K12 = np.array(X_MAS_ERR_K12), np.array(Y_MAS_ERR_K12)


SEP_H23, PA_H23 = np.array(SEP_H23), np.array(PA_H23)
SEP_K12, PA_K12 = np.array(SEP_K12), np.array(PA_K12)
SEP_ERR_H23, PA_ERR_H23 = np.array(SEP_ERR_H23), np.array(PA_ERR_H23)
SEP_ERR_K12, PA_ERR_K12 = np.array(SEP_ERR_K12), np.array(PA_ERR_K12)

NAME_H23, NAME_K12 = np.array(NAME_H23), np.array(NAME_K12)

## palette colors ##

## palette H23 ##
# if one want to create a palette of colours
n_colors = 27
# list of the colors for the palette
palette = cm.get_cmap('gist_rainbow', n_colors)
arg_palette = np.arange(0,n_colors,1)
# let's shuffle it
random.shuffle(arg_palette)
palette_not_shuffled = palette(arg_palette)
palette_shuffled = palette(arg_palette)

# special palette for Super-Earth survey in H23
palette = np.array([
       [1.        , 0.04573805, 0.        , 1.        ],
       [0.        , 1.        , 0.33085194, 1.        ],
       [0.8138796 , 0.        , 1.        , 1.        ],
       [0.        , 1.        , 0.95119934, 1.        ],
       [0.90      , 0.        , 0.        , 1.        ], # red
       [0.        , 0.4222408 , 1.        , 1.        ],
       [1.        , 0.46153846, 0.        , 1.        ],
       [0.        , 1.        , 0.74441687, 1.        ],
       [0.95      , 0.9        , 0.53763441, 1.        ],
       [0.6229097 , 0.        , 1.        , 1.        ],
       [1.        , 0.66943867, 0.        , 1.        ],
       [0.95      , 0.7       , 0.        , 1.        ], # orange
       [1.        , 0.        , 0.16      , 1.        ],
       [0.        , 0.0041806 , 1.        , 1.        ],
       [0.        , 0.2132107 , 1.        , 1.        ],
       [0.        , 0.840301  , 1.        , 1.        ],
       [0.2048495 , 0.        , 1.        , 1.        ],
       [1.        , 0.25363825, 0.        , 1.        ],
       [1.        , 0.        , 0.12406948, 1.        ],
       [1.        , 0.        , 0.9590301 , 1.        ],
       [1.        , 0.        , 0.75      , 1.        ],
       [0.98316008, 0.        , 0.        , 1.        ],
       [0.9989605 , 0.        , 0.        , 1.        ],
       [0.99106029, 0.        , 0.        , 1.        ],
       [0.        , 0.6312709 , 1.        , 1.        ],
       [1.        , 0.87733888, 0.        , 1.        ],
       [0.8319398 , 0.        , 1.        , 1.        ]])


# if one want to give order between the targets while plotting them
# basic
cc_zorder_details = np.arange(n_colors,0,-1)*100
# specifications for Super-Earth survey
cc_zorder_details = np.array([2700, 2600, 2500, 2400, 230, 2200, 2100, 2000, 1900,
       180, 170, 160, 1500, 1400, 1300, 1200, 1100, 10000,  900,  800,
        700,  600,  500,  400,  300,  200,  100])


## palette K12 ##
palette_K12 = np.array([[0.9       , 0.        , 0.        , 1.        ],
                        [0.9       , 0.        , 0.        , 1.        ],
                        [0.95      , 0.9       , 0.53763441, 1.        ],
                        [0.95      , 0.9       , 0.53763441, 1.        ],
                        [0.95      , 0.9       , 0.53763441, 1.        ],
                        [0.95      , 0.9       , 0.53763441, 1.        ]])
                        
                        
## palette isochrone (1 color = 1 isochrone) ##
n_colors = 5
palette_evol_CMD = cm.get_cmap('gist_rainbow', n_colors)
arg_palette = np.arange(0,n_colors,1)
palette_evol_CMD = palette_evol_CMD(arg_palette)


# list of the colors for the palette isochrone - model (1 color = 1 model or 1 filter)
n_colors = 7
palette_model_filter = cm.get_cmap('gist_rainbow', n_colors)
arg_palette_model_filter = np.arange(0,n_colors,1)
color_model_filter = palette_model_filter(arg_palette_model_filter)
