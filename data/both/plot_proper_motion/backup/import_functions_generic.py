"""
NAME:
 import_functions_generic

PURPOSE:
 Import functions usually used

INPUTS:
 none

OUTPUTS:
 none

MODIFICATION HISTORY:
 21/10/01, Written by Célia DESGRANGE, IPAG/MPIA
 22/01/06, Modified by Célia DESGRANGE, added function pd_string2float
 22/01/13, Modified by Célia DESGRANGE,
    .updated function index_closer (can now take several constraint)

CONTRIBUTION:
 Arthur Vigan: function legend_without_duplicate_labels (from its CMD routine)
"""

from import_packages_generic import *

## General functions to make nice plot ##
def mise_en_page(ax,x_min_step=0.2,x_maj_step=1,y_maj_step=1,y_min_step=0.2, x_step=0, y_step=0, grid_maj=0, grid_min=0):
    ax.tick_params(axis='x',which='major', width=1.3,length=8)
    ax.tick_params(axis='x',which='minor', width=1,length=5)
    ax.tick_params(axis='y',which='major', width=1.3,length=8)
    ax.tick_params(axis='y',which='minor', width=1,length=5)

    if x_step :
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_maj_step))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(x_min_step))
    if y_step :
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_maj_step))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(y_min_step))
    if grid_maj :
        ax.grid(which='major',color=[0.5,0.5,0.5],ls='-')
    if grid_min :
        ax.grid(which='minor',color=[0.8,0.8,0.8],ls='-')
    return

#else : ax.plot(X,Y,color=c,ls=ls,ms=10,label=lab,lw=lw)


def fais_moi_un_subplot(ax,X,Y,labx='',laby='',tit='',xlim=None,ylim=None,xlog=False,ylog=False,
                        xti_old=None,xti_new=None,yti_old=None,yti_new=None,dx=0,dy=0,grid_maj=0,
                        superpo=False,lab='',leg=False,loc=0,c=[0,0,1],lw=1,ls='dotted',siz='3',mkr=None):

    if dx != 0 or dy != 0 :
        ax.errorbar(X,Y,dy,dx,mkr,markersize=siz,label=lab,c=c)
    elif mkr != None :
        ax.plot(X,Y,color=c,ls=ls,marker=mkr,ms=siz,label=lab,lw=lw)
    else :
        ax.plot(X,Y,color=c,ls=ls,ms=10,label=lab,lw=lw)
    if xti_new != xti_old:
        plt.setp(ax, xticks=xti_old, xticklabels=xti_new)
    if yti_new != yti_old:
        plt.setp(ax, yticks=yti_old, yticklabels=yti_new)
    if xlog : ax.set_xscale('log')
    if ylog : ax.set_yscale('log')
    if leg : plt.legend(loc=loc)
    if xlim != None : ax.set_xlim(xlim)
    if ylim != None : ax.set_ylim(ylim)
    mise_en_page(ax,x_min_step=0.2,x_maj_step=1,y_maj_step=1,y_min_step=0.2, x_step=0, y_step=0, grid_maj=grid_maj, grid_min=0)
    ax.set_xlabel(labx)
    ax.set_ylabel(laby)
    ax.set_title(tit)
    return


def legend_without_duplicate_labels(ax,loc=0,numpoints=1,ncol=1,frameon=True):
    '''
    Remove duplicate labels in legend.
    '''
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),numpoints=numpoints,ncol=ncol,loc=loc,frameon=frameon)


def define_color_shades(list_colors, n_colors, show = 0, blend = 0, light = 1):
    '''
    Define color shades (for example in maps).

    blend = 1 if list_colors is a list of strings
    '''
    if blend :
        palette = seaborn.blend_palette(list_colors, n_colors) # palette
        cmap = seaborn.blend_palette(list_colors, n_colors, as_cmap=True) # cmap
    else : # light
        palette = seaborn.light_palette(list_colors, n_colors) # palette
        cmap = seaborn.light_palette(list_colors, n_colors, as_cmap=True) # cmap

    arg_cmap = np.linspace(1,0,n_colors,endpoint=True)
    colors = cmap(arg_cmap) # list
    if show :
        display(palette)
        display(cmap)
        print(colors)
    return palette,cmap,colors


## generic function image ##
def distance(x,y,x0,y0,pixarc=12.25):
    return np.sqrt((x-x0)**2+(y-y0)**2)*pixarc

##generic function int/float ##
def round_sig(X, sig=2):
    if type(X) == int or type(X) == float :
        return round(X, sig-int(floor(log10(abs(X))))-1)
    else :
        L = []
        for k in range(len(X)):
            x = X[k]
            L.append(round(x, sig-int(floor(log10(abs(x))))-1))
        return np.array(L)


## generic function list ##
def index_closer(wanted,L,constraints='one',output_diff=False):
    '''
    Inputs:
        'wanted' (type = int/float or 1D-list/array): int/float if 'constraints' set to 'one'
        'L' (type = 1D-list/array or 2D-list/array): 1D if 'constraints' set to 'one')
        (optional)
        'constraints' (type = string): if 'one', simpler case, there is only one constraint,
            otherwise, 'several'.
        'output_diff' (type = boolean): if True, return an additional output corresponding
            to the difference between the 'wanted' value and the one closest in the list 'L'.
    Outputs:
        np.argmin(diff) (type = int/float): returns the index giving the value L[index][:] closer
        to the 'wanted' value for one or several constraints.

    # example
    Inputs:
        wanted = [27,8]
        L = [[20,30,45],[2,7,4]]
    Outputs:
         np.argmin(diff) = 1
         (optional)
         diff = [13  4 22]

    '''
    if constraints == 'one':
        diff = np.abs(L-wanted)
    else :
        # wanted list of size Nparam
        Nparam = len(wanted)
        # L should be a 2D-array of size Nparam*whatever
        L = np.array(L)
        DIFF = []
        for k in range(Nparam):
            diff = np.abs(L[k]-wanted[k])
            DIFF.append(diff)
        diff = np.sum(DIFF,axis=0)
    if output_diff : return np.argmin(diff), diff
    else : return np.argmin(diff)



## generic function string ##
def select_string_between_characs(string,charac_deb,charac_fin,number_apparition=0):
    apparition=0
    string_of_interest = select_string_after_charac(string,charac_deb,number_apparition)
    #print(string_of_interest)
    return select_string_before_charac(string_of_interest,charac_fin)


def select_string_before_charac(string,charac,number_apparition=0):
    '''
    Return the string before the character "charac".
    By default, return the string "string" localized before the first apparition of the character "charac", but in option it can be before the number of apparition "number_apparition" (and thus containing the character "charac" "number_apparition" times).
    By default, the number of apparition is counted from the beginning of the string, but it could be also counted from the end of the string, if "number_apparition" is negative.
    '''
    if number_apparition < 0 :
        number_apparition = string.count(charac) + number_apparition
    apparition=0
    for i in range(len(string)):
        if charac == string[i]:
            if apparition == number_apparition:
                #print(apparition)
                return string[:i]
            apparition+=1
    return 'There is not the character "{}" in the string given.'.format(charac)

def select_string_after_charac(string,charac,number_apparition=0):
    '''
    Return the string after the character "charac".
    By default, return the string "string" localized after the first apparition of the character "charac", but in option it can be after the number of apparition "number_apparition".
    By default, the number of apparition is counted from the beginning of the string, but it could be also counted from the end of the string, if "number_apparition" is negative.
    '''
    if number_apparition < 0 :
        number_apparition = string.count(charac) + number_apparition
    apparition=0
    for i in range(len(string)):
        if charac == string[i]:
            if apparition == number_apparition:
                #print(apparition)
                return string[i+1:]
            apparition+=1
    return 'There is not the character "{}" in the string given.'.format(charac)

## convert string -> float ##, for instance for files extractions
def pd_string2float(string,sep):
    '''
    Convert string -> float.
    In the origin, function used when extracting numbers from tables.
    Inputs:
        .'string' (type = string or list/array of strings): number written with a string type
        .'sep' (type = string): separation
    Output:
        .'L' (type = float or array of floats): of the input string
    '''
    L = []
    try :
        return float(string[1:-1].split(sep))
    except :
        n = len(string[1:-1].split(sep))
        for i in range(n):
            L.append(float(string[1:-1].split(sep)[i]))
        return np.array(L)


## interpolation ##
def fct_interpolation(x,y):
    '''Interpolate the data y=spl(x) with a 1-D smoothing spline spl and input values x.
    Return the function of interpolation spl.'''
    return interpolate.UnivariateSpline(x,y)

def mass2temperature(mass,temp,mass_value):
    '''
    Input: mass (= list x values), temp (= list y values) and mass_value that one wants to convert in temperature.
    Output: temp_value corresponding to f(mass_value)
    '''
    f = fct_interpolation(mass,temp)
    return f(mass_value)




# Functions to convert easily units (one way)
## convert sep2au etc
def sep2au(sep,d=10):
    '''
    Convert angular separation 'sep' (in arcsecond) to astrononomical unit
    for a system located at a distance  (in parsec).
    Note: d_object is a parameter defined by the class 'Object'.
    The user should update the parameters of the class 'Object' accordingly.
    '''
    return sep*d

def convert_ax_sep_to_au(ax):
    '''
    Add to the already defined axis 'ax' a second horizontal axis corresponding
    to the distance in astronomical unit.
    Require the funtion 'sep2au'.
    '''
    y1, y2 = ax.get_xlim()
    ax1.set_xlim(sep2au(y1), sep2au(y2))
    ax1.figure.canvas.draw()
    return

def contr2mag(contr):
    '''
    Convert contrast 'contr' to magnitude.
    '''
    return -2.5*np.log10(contr)

def convert_ax_contr_to_mag(ax):
    '''
    Add to the already defined axis 'ax' a second vertical axis corresponding
    to the magnitude.
    Require the funtion 'contr2mag'.
    '''
    x1, x2 = ax.get_ylim()
    ax2.set_ylim(contr2mag(x1), contr2mag(x2))
    ax2.figure.canvas.draw()
    return

# Functions to convert easily units (the other way)
def au2sep(au,d=10):
    '''
    Convert astrononomical unit to angular separation 'sep' (in arcsecond)
    for a system located at a distance 'd_object' (in parsec).
    Note: d_object is a parameter defined by the class 'Object'.
    The user should update the parameters of the class 'Object' accordingly.
    '''
    return au/d

def convert_ax_sep_to_au(ax):
    '''
    Add to the already defined axis 'ax' a second horizontal axis corresponding
    to the distance in astronomical unit.
    Require the funtion 'au2sep'.
    '''
    y1, y2 = ax.get_xlim()
    ax1.set_xlim(au2sep(y1), au2sep(y2))
    ax1.figure.canvas.draw()
    return

def mag2contr(mag):
    '''
    Convert magnitude 'mag' to contrast.
    '''
    return 10**(-mag/2.5)

def convert_ax_mag_to_contr(ax):
    '''
    Add to the already defined axis 'ax' a second vertical axis corresponding
    to the contrast.
    Require the funtion 'mag2contr'.
    '''
    x1, x2 = ax.get_ylim()
    ax2.set_ylim(mag2contr(x1), mag2contr(x2))
    ax2.figure.canvas.draw()
    return


# Additional functions to convert easily units - might not be useful here
def parallax2parsec(parallax):
    '''
    Convert parallax 'parallax' (in arcseconds) of a given system to its distance (in parsecs).
    '''
    return 1/parallax
