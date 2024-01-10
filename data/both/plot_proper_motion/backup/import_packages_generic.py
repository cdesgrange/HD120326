"""
NAME:
 import_packages_generic

PURPOSE:
 Import packages usually used

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 21/10/01, Written by Célia DESGRANGE, IPAG/MPIA
"""

# general
import numpy as np
from glob import glob
from pathlib import Path
import subprocess
import time
import argparse
import os

# astro
from astropy.io import ascii, fits
from astropy import constants as const
from astropy.table import Table
from astropy import units as u
import astropy.constants as cst
import astropy.units as units

# math
from math import log10, floor
import cmath
import random
from scipy import optimize, interpolate, ndimage
from scipy.interpolate import RectBivariateSpline as rbs

# dataframe
import pandas as pd

# graphic
import seaborn

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, NullLocator, FormatStrFormatter, AutoMinorLocator,AutoLocator,LogFormatter,LogLocator,MaxNLocator)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.axes import Axes
from matplotlib import colors, cm
import matplotlib as mpl
from matplotlib.patches import Arrow, Circle, ArrowStyle, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib.axes import rcParams
plt.style.use('classic')
mpl.rc('image', cmap='magma')#, origin='lower')
#mpl.rc('text', usetex=True)
#mpl.rc(mathtext.fontset = 'stix')


rcParams.update({'font.size': 14,  'axes.labelsize' : 14, 'legend.fontsize' : 12,
"font.family": "serif", 'text.usetex' : True, "font.serif": [], "font.sans-serif": [],
'legend.handlelength': 1.3, 'legend.borderaxespad' : 0.8, 'legend.columnspacing' : 0.5, 'legend.handletextpad' : 0.5})
            
rcParams['mathtext.fontset'] = 'stix'
