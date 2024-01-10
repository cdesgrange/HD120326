# grater_diskfm_mcmc

This git depository is for combining Julien's Grater version of the code + vip_hci + pyklip-diskFM + an MCMC for the analysis of disk in Sopia's PhD.

**Installation**

Install Miniconda:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

Create an environment for this project:
```
conda create --name GraterVIPdiskFM-env python=3.7 ipython
conda activate GraterVIPdiskFM-env
```

Install VIP: `pip install vip_hci`
Then update emcee package with the latest version:`pip install --upgrade emcee`


Install the rest of the needed packages:
```
pip install git+https://bitbucket.org/pyKLIP/pyklip.git
pip install h5py 
pip install git+https://github.com/ericmandel/pyds9.git#egg=pyds9
```





