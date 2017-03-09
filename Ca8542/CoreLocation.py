from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import os
import vhsDataHandler as vhs
import LimbFinding as LF
  
def CalciumCoreRegion(fluxarray):
    """NAME: CoreRegion 
    PURPOSE: Returns an array for the indices in the region of the calcium absorption line given an array of fluxes.
    INPUTS:  fluxarray       - The array of flux values
    OUTPUTS: xs              - The array of indices in the region around the absorption line
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    stokesslice=vhs.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=vhs.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.CoreRegion(flux)
            In [7]: xs
            Out[7]: array([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    """
    fluxarray=np.array(fluxarray)
    flux20=fluxarray[20]
    neighborhood1=fluxarray[48:76]
    neighborhood2=fluxarray[48:73]
    relmins=ss.argrelextrema(neighborhood1, np.less)[0]+48
    relmax=ss.argrelextrema(neighborhood2, np.greater)[0]+48
    if len(relmins)==3:
        roughcenter=relmins[1]
    elif len(relmins)==1:
        roughcenter=relmins[0]
    elif len(relmins)==0:
        roughcenter=np.argmax(neighborhood1)+48
    elif len(relmins)==2:
        if relmins[0]>57 and relmins[0]<63:
            roughcenter=relmins[0]
        elif relmins[1]>57 and relmins[1]<63:
            roughcenter=relmins[1]
        elif len(relmax)==2:
            roughcenter=np.int((relmax[1]-relmax[0])/2+relmax[0])
        elif len(relmax)==1:
            roughcenter=np.int((relmins[1]-relmax[0])/2+relmax[0])
        else:
            roughcenter=60
    else:
        roughcenter=np.argmin(neighborhood1)+48
    neighborhoodmin=fluxarray[roughcenter]
    if neighborhoodmin/flux20<.4:
        xs=np.arange(roughcenter-7, roughcenter+8)
    elif neighborhoodmin/flux20<.5:
        xs=np.arange(roughcenter-4, roughcenter+5)
    elif neighborhoodmin/flux20<.8:
        xs=np.arange(roughcenter-2, roughcenter+3)        
    else:
        xs=np.arange(roughcenter-1, roughcenter+2)        
    return xs

def WaterCoreRegion(fluxarray):
    """NAME: CoreRegion 
    PURPOSE: Returns an array for the indices in the region of the terrestrial water absorption line located at 8540.7959 Angstroms given an array of fluxes.
    INPUTS:  fluxarray       - The array of flux values
    OUTPUTS: xs              - The array of indices in the region around the absorption line
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    stokesslice=vhs.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=vhs.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.CoreRegion(flux)
            In [7]: xs
            Out[7]: array([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    """
    neighborhood1=fluxarray[14:29]
    neighborhoodmin=np.min(neighborhood1)
    roughcenter=np.argmin(neighborhood1)+14 
    comparison=fluxarray[roughcenter-5]
    if neighborhoodmin/comparison>0.92:
        xs=np.arange(roughcenter-4, roughcenter+3)
    else:
        xs=np.arange(roughcenter-3, roughcenter+3)
    return xs

def PolyMinLoc(fluxarray, xs, deg=2):
    """NAME: PolyMinLoc 
    PURPOSE: Uses a polynomial fit to find a more precise location for the minimum of the absorption line.
    INPUTS:  fluxarray       - The array of flux values
             xs              - The array of indices in the region around the absorption line
    OUTPUTS: loc             - The location of the minimum of the polynomial fit of the absorption to two decimals.
    EXAMPLE:
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    stokesslice=vhs.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=vhs.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.CoreRegion(flux)
                    polyminloc=CL.PolyMinLoc(flux, xs)
            In [7]: polyminloc
            Out[7]: 61.789999999999999
    """
    ys=fluxarray[xs[0]:xs[-1]+1]
    px=np.polyfit(xs, ys, deg)
    p=np.poly1d(px)
    x=np.linspace(xs[0],xs[-1], 1000)
    y=p(x)
    minloc=np.argmin(y)
    loc=np.abs(xs[-1]-xs[0])*(minloc/1000)+xs[0]
    return loc

def LocChecker(datapath, row, col, deg=2, line='Calcium'):
    """NAME: LocChecker 
    PURPOSE: Displays the spectrum for a specific pixel, the polynomial fit to the absorption line, and the minimum of the polynomial.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             row            - The row which is being operated upon.
             col            - The column which is being operated upon.
             line           - The absorption line to look at: Calcium or Water
    OUTPUTS: N/A            - Returns nothing. Only shows the matplotlib plot.
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    CL.LocChecker(datapath=datapath, row=1500, col=340)
    """
    flux=vhs.Spectrum(datapath=datapath, stokes='I', slicenumber=vhs.SliceNumber(row), xpos=col)
    if line=='Calcium':
        xs=CalciumCoreRegion(flux)
    elif line=='Water':
        xs=WaterCoreRegion(flux)
    ys=flux[xs[0]:xs[-1]+1]
    px=np.polyfit(xs, ys, deg)
    p=np.poly1d(px)
    x=np.linspace(xs[0],xs[-1], 1000)
    y=p(x)
    minloc=np.argmin(y)
    loc=np.abs(xs[-1]-xs[0])*(minloc/1000)+xs[0]
    fig=plt.figure()
    plt.plot(flux)
    plt.plot(xs, p(xs))
    plt.scatter(loc, p(loc))
    plt.title("Row: "+str(row)+", Col: "+str(col)+", Degree of Poly Fit: "+str(deg))
    plt.annotate('min loc = '+str(loc), xy=(loc, p(loc)), xytext=(loc, p(loc)-5000))
    plt.show()

def PixelIndices(slicedata, array, row, col, line='Calcium'):
    """NAME: PixelIndices
    PURPOSE: Assigns the value to the element of the array of index locations for a pixel.
    INPUTS:  slicedata      - The array of intensities for a horizontal slice of the sun.
             array          - The array which will hold the index information.
             row            - The row which is being operated upon.
             col            - The column which is being operated upon.
             line           - The absorption line to look at: Calcium or Water
    OUTPUTS: N/A            - Only reassigns the value of the array element.
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    stokesslice=vhs.SingleStokes(datapath, 'I', slicenumber=400)
                    indexmap=np.zeros((len(disk), len(disk)))
                    CL.PixelIndices(stokesslice, indexmap, 400, 700)
                    polyminloc=CL.PolyMinLoc(flux, xs)
            In [7]: indexmap[400][700]
            Out[7]: 61.789999999999999
    """
    flux=vhs.LoadedDataSpectrum(slicedata=slicedata, xpos=col)
    if line=='Calcium':
        xs=CalciumCoreRegion(flux)
    elif line=='Water':
        xs=WaterCoreRegion(flux)
    loc=PolyMinLoc(flux, xs)
    array[row][col]=loc
    
def IndexMap(datapath, array, line='Calcium'):
    """NAME: IndexMap 
    PURPOSE: Given a folder of FITS files and an array of full disk intensity information, return an array containing the index of the core of the absorption line.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             array          - The array of full-disk intensity information.
             line           - The absorption line to look at: Calcium or Water
    OUTPUTS: indexmap       - The array containing the indices of the core of the absorption line for each pixel.
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    indexmap=CL.IndexMap(datapath, disk[1020:1028])
            In [7]: indexmap[4][29:35]
            Out[7]: array([  0.   ,   0.   ,  63.016,  60.264,  61.748,  61.472])
    """
    indexmap=np.zeros((len(array), len(array[0])))
    solarboolean=LF.ContainsSun(datapath=datapath, disk=array)
    gapcols=LF.GapCols(datapath)
    limbs=LF.LimbIndices(datapath, array, solarboolean)
    for i in range(len(array)):
        if solarboolean[i]==True:
            slicedata=vhs.SingleStokes(datapath=datapath, stokes='I', slicenumber=i)
            (limb1, limb2)=limbs[i]
            j=limb1
            while(j<gapcols[0]):
                PixelIndices(slicedata, indexmap, i, j, line=line)
                j=j+1
            j=gapcols[1]
            while(j<limb2):
                PixelIndices(slicedata, indexmap, i, j, line=line)
                j=j+1
    return indexmap

def SaveIndexMap(datapath, disk, line='Calcium', outputtype='.fits', outputpath='local'):
    """NAME: IndexMap 
    PURPOSE: Given a folder of FITS files and an array of full disk intensity information, return an array containing the index of the core of the absorption line.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             array          - The array of full-disk intensity information.
             line           - The absorption line to look at: Calcium or Water
    OUTPUTS: indexmap       - The array containing the indices of the core of the absorption line for each pixel.
    EXAMPLE: 
            In [6]: datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/FullDiskPlots /k4vhs160409t174101_oid114602236001741_cleanedI80fulldisk.fits')
                    indexmap=CL.IndexMap(datapath, disk[1020:1028])
            In [7]: indexmap[4][29:35]
            Out[7]: array([  0.   ,   0.   ,  63.016,  60.264,  61.748,  61.472])
    """        
    im=IndexMap(datapath=datapath, array=disk, line=line)
    filen=datapath[0:18]
    if outputpath=='local':
        baseoutput='/Users/jhamer/Research/Output'
        outputpath=baseoutput+'/'+str(filen)+'/IndexMaps'
    else:
        outputpath=outputpath
    if os.path.isdir(outputpath)==False:
        os.mkdir(outputpath)
    if outputtype=='.fits':
        vhs.ArraytoFITSImage(array=im, outputpath=outputpath, outputname=str(datapath)+str(line)+'IndexMap' )
    elif outputtype=='visual':
        plt.clf()
        fig=plt.imshow(im, origin='lower')
        plt.title(str(line)+' Index Map')
        plt.show(fig)
    else:
        fig=plt.imshow(im)
        plt.colorbar()
        fn=str(filen)+str(line)+'IndexMap'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title(str(line)+' Index Map')
        plt.savefig(fullpath)
        plt.close()