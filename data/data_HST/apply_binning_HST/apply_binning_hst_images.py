#!/usr/bin/python
import numpy as np
print('hey')
from functions_binning import *
print('=== Bin images ===')

# Define saving directory
L = time.localtime()
date = "{}-{}-{}-{}h{}min{}s".format(L[0],L[1],L[2],L[3],L[4],L[5],L[6])
SAVING_DIR = 'ouputs/'+date+'/'



os.makedirs(SAVING_DIR,exist_ok=True)

# Specify files to be binned
FNs =  ['IM12_derot_mask+spiders','IM21_derot_mask+spiders',
        'IM_cadi_derot', 'im_obs_mask_max_final', 'im_obs_mask_min_final'
        ]

FNs = ['IM12_derot_snr', 'IM12_derot', 'IM12_derot_mask+spiders_snr', 'IM12_derot_mask+spiders']


folder = 'results_run/'
folder = ''

BFs = np.arange(2,6,1) # binning factor

fc = np.nanmean # function use to bin, mean, sum..

for i in range(len(FNs)):
    for j in range(len(BFs)):
        fn, bf = FNs[i], BFs[j]
        print('The file "{}" is binned with a factor {}.'.format(fn, bf))
        f = fits.getdata('data/'+folder+fn+'.fits')
        block_reduce_im(f, bf, writeto=True, saving_dir=SAVING_DIR, namesave=fn+'_bf'+str(bf), func=fc)
