from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage.filters as filters


def GapCols(datapath, slicenumber='0000'):
    """NAME: GapCols
    PURPOSE: Returns the columns of the beginning and end of the gap for a FITs.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             slicenumber     - The vertical slicenumber from 0000 to 2047 - always a string of 4 digits!
    OUTPUTS: gapcols         - A tuple of [gapcol1, gapcol2]
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    gapcols=LF.GapCols(datapath=datapath)
            In [7]: gapcols
            Out[7]: [981, 1045]
    """
    basepath='/Users/jhamer/Research/Data'
    prefix=basepath+'/'+str(datapath)
    filen=datapath[0:18]
    hdr=fits.getheader(prefix+'/'+str(filen)+'_'+slicenumber+'.fts.gz')
    gapcols=[hdr['gapcol1'], hdr['gapcol2']]
    return gapcols

def MaxGradLoc(array):
    """NAME: MaxGradLoc
    PURPOSE: Returns the index of maximal gradient in an array.
    INPUTS:  array         - The  array to perform gradient on.
    OUTPUTS: index         - The index where the gradient is maximal.
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    gapcols=LF.GapCols(datapath=datapath)
                    maxloc=LF.MaxGradLoc(disk[1000][0:gapcols[0]])
            In [7]: maxloc
            Out[7]: 28
    """
    grad=np.gradient(array)
    index=np.argmax(grad)
    return index

def MinGradLoc(array):
    """NAME: MinGradLoc
    PURPOSE: Returns the index of minimal gradient in an array.
    INPUTS:  array         - The  array to perform gradient on.
    OUTPUTS: index         - The index where the gradient is minimal.
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    gapcols=LF.GapCols(datapath=datapath)
                    minloc=LF.MinGradLoc(disk[1000][gapcols[1]:2010])+gapcols[1]
            In [7]: minloc
            Out[7]: 2002
    """
    grad=np.gradient(array)
    index=np.argmin(grad)
    return index

def RowLimbIndices(datapath, row):
    """NAME: LimbIndices
    PURPOSE: Returns the indices of the left and right limbs of the Sun in a row.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             row             - The row array of intensity values to find the limbs of.
    OUTPUTS: limb1           - The index of the left limb
             limb2           - The index of the right limb
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    limb1, limb2=LF.LimbIndices(datapath, disk[789])
            In [7]: limb1, limb2
            Out[7]: (43, 1990)
    """
    gapcols=GapCols(datapath=datapath)
    row1=row[0:gapcols[0]]
    row2=row[gapcols[1]:2020]
    limb1=MaxGradLoc(row1)
    limb2=MinGradLoc(row2)
    limb2=limb2+gapcols[1]
    return limb1, limb2
    
def ContainsSun(datapath, disk):
    """NAME: ContainsSun
    PURPOSE: Checks that the gradient of the intensity reaches a certain threshold to determine if the row contains the disk of the sun. Returns a boolean array.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             disk            - The array of arrays of intensity values.
    OUTPUTS: solarboolean    - The boolean array. True when a row contains the solar disk.
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    solarbool=LF.ContainsSun(datapath, disk)
            In [7]: solarbool
            Out[7]: array([False, False, False, ..., False, False, False], dtype=bool)
            In [8]: solarbool[40:44]
            Out[8]: array([False, False,  True,  True], dtype=bool)
In [ ]:

    """
    solarboolean=[]
    for i in range(len(disk)):
        row=disk[i][0:GapCols(datapath=datapath)[0]]
        grad=np.gradient(row)
        if np.max(grad) < 100:
            solarboolean.append(False)
        elif np.max(grad) > 100:
            solarboolean.append(True)
    solarboolean=np.array(solarboolean, dtype=bool)
    return solarboolean

def LimbIndices(datapath, array, solarbool):
    '''NAME: LimbIndices
    PURPOSE: Returns an array consisting of the indices of the left and right limbs of the Sun for each row of the array.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             array           - The array of intensities for the full disk of the sun
             solarbool       - The boolean array returned by ContainsSun
    OUTPUTS: limbs           - A len(array)x2 array where each 2-tuple is the leftand right limb of the sun in each row.
    EXAMPLE:
    '''
    smoothed_array=filters.gaussian_filter(array, (1,10))
    gapcols=GapCols(datapath)
    limbs=[]
    width=200
    i=0
    if solarbool[i]==False:
        while(solarbool[i]==False):
            limbs.append([gapcols[0], gapcols[1]])
            i=i+1
            if i+1==len(array):
                break
    row=smoothed_array[i]
    row1=row[0:gapcols[0]]
    row2=row[gapcols[1]:2010]
    limb1i=MaxGradLoc(row1)
    limb2i=MinGradLoc(row2)+gapcols[1]
    while(solarbool[i]==True):
        row=smoothed_array[i]
        if limb1i-width>=0 and limb1i+width<=gapcols[0]:
            row1=row[limb1i-width:limb1i+width]
            limb1=MaxGradLoc(row1)
            limb1=limb1+limb1i-width
        elif limb1i-width<0:
            row1=row[0:limb1i+width]
            limb1=MaxGradLoc(row1)
            limb1=limb1
        elif limb1i+width>gapcols[0]:
            row1=row[limb1i-width:gapcols[0]]
            limb1=MaxGradLoc(row1)
            limb1=limb1+limb1i-width
        if limb2i-width>=gapcols[1] and limb2i+width<=2010:
            row2=row[limb2i-width:limb2i+width]
            limb2=MinGradLoc(row2)
            limb2=limb2+limb2i-width
        elif limb2i-width<gapcols[1]:
            row2=row[gapcols[1]:limb2i+width]
            limb2=MinGradLoc(row2)
            limb2=limb2+gapcols[1]
        elif limb2i+width>2010:
            row2=row[limb2i-width:2010]
            limb2=MinGradLoc(row2)
            limb2=limb2+limb2i-width
        limb1i=limb1
        limb2i=limb2
        limbs.append([limb1, limb2])
        i=i+1
        if i+1==len(array):
            break
    while(solarbool[i]==False and i<2047):
        limbs.append([gapcols[0], gapcols[1]])
        i=i+1
        if i+1==len(array):
            break
    if solarbool[i]==False:
        limbs.append([gapcols[0], gapcols[1]])
    elif solarbool[i]==True:
        row=smoothed_array[i]
        row1=row[0:gapcols[0]]
        row2=row[gapcols[1]:2010]
        limb1=MaxGradLoc(row1)
        limb2=MinGradLoc(row2)+gapcols[1]
        limbs.append([limb1, limb2])
    return limbs
