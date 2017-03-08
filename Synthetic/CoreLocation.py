from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import os
import v9sDataHandler as v9s
import LimbFinding as LF
from datetime import datetime

  
def IronCoreRegion(fluxarray):
    """NAME: IronCoreRegion 
    PURPOSE: Returns an array for the indices in the region of the FeI 6302.4995 absorption line given an array of fluxes.
    INPUTS:  fluxarray       - The array of flux values
    OUTPUTS: xs              - The array of indices in the region around the absorption line
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    stokesslice=v9s.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=v9s.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.IronCoreRegion(flux)
            In [7]: xs
            Out[7]: array([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    """
    fluxarray=np.array(fluxarray)
    flux20=fluxarray[20]
    neighborhood1=fluxarray[76:86]
    roughcenter=np.argmin(neighborhood1)+76
    neighborhoodmin=fluxarray[roughcenter]
    xs=np.arange(roughcenter-2, roughcenter+3)        
    return xs

def Oxygen1CoreRegion(fluxarray):
    """NAME: Oxygen1CoreRegion 
    PURPOSE: Returns an array for the indices in the region of the terrestrial oxygen absorption line located at 6302.0005 Angstroms given an array of fluxes.
    INPUTS:  fluxarray       - The array of flux values
    OUTPUTS: xs              - The array of indices in the region around the absorption line
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    stokesslice=v9s.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=v9s.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.Oxygen1CoreRegion(flux)
            In [7]: xs
            Out[7]: array([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    """
    neighborhood1=fluxarray[55:70]
    neighborhoodmin=np.min(neighborhood1)
    roughcenter=np.argmin(neighborhood1)+55
    comparison=fluxarray[roughcenter-5]
    xs=np.arange(roughcenter-1, roughcenter+2)
    return xs

def Oxygen2CoreRegion(fluxarray):
    """NAME: Oxygen2CoreRegion 
    PURPOSE: Returns an array for the indices in the region of the terrestrial oxygen absorption line located at 6302.7627 Angstroms given an array of fluxes.
    INPUTS:  fluxarray       - The array of flux values
    OUTPUTS: xs              - The array of indices in the region around the absorption line
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    stokesslice=v9s.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=v9s.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
                    xs=CL.Oxygen2CoreRegion(flux)
            In [7]: xs
            Out[7]: array([55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    """
    neighborhood1=fluxarray[89:97]
    neighborhoodmin=np.min(neighborhood1)
    roughcenter=np.argmin(neighborhood1)+89
    comparison=fluxarray[roughcenter-5]
    xs=np.arange(roughcenter-1, roughcenter+2)
    return xs

def PolyMinLoc(fluxarray, xs, deg=2):
    """NAME: PolyMinLoc 
    PURPOSE: Uses a polynomial fit to find a more precise location for the minimum of the absorption line.
    INPUTS:  fluxarray       - The array of flux values
             xs              - The array of indices in the region around the absorption line
    OUTPUTS: loc             - The location of the minimum of the polynomial fit of the absorption to two decimals.
    EXAMPLE:
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    stokesslice=v9s.SingleStokes(datapath, 'I', slicenumber=400)
                    flux=v9s.LoadedDataSpectrum(slicedata=stokesslice, xpos=700)
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

def LocChecker(datapath, row, col, deg=2, line='Iron'):
    """NAME: LocChecker 
    PURPOSE: Displays the spectrum for a specific pixel, the polynomial fit to the absorption line, and the minimum of the polynomial.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             row            - The row which is being operated upon.
             col            - The column which is being operated upon.
             line           - The absorption line to look at: Calcium or Oxygen
    OUTPUTS: N/A            - Returns nothing. Only shows the matplotlib plot.
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    CL.LocChecker(datapath=datapath, row=1500, col=340)
    """
    flux=v9s.Spectrum(datapath=datapath, stokes='I', slicenumber=v9s.SliceNumber(row), xpos=col)
    if line=='Iron':
        xs=IronCoreRegion(flux)
    elif line=='Oxygen1':
        xs=Oxygen1CoreRegion(flux)
    elif line=='Oxygen2':
        xs=Oxygen2CoreRegion(flux)
    ys=flux[xs[0]:xs[-1]+1]
    px=np.polyfit(xs, ys, deg)
    p=np.poly1d(px)
    x=np.linspace(xs[0],xs[-1], 1000)
    y=p(x)
    minloc=np.argmin(y)
    loc=np.abs(xs[-1]-xs[0])*(minloc/1000)+xs[0]
    fig=plt.figure()
    plt.plot(flux)
    plt.plot(x, p(x))
    plt.scatter(loc, p(loc))
    plt.title("Row: "+str(row)+", Col: "+str(col)+", Degree of Poly Fit: "+str(deg))
    plt.annotate('min loc = '+str(loc), xy=(loc, p(loc)), xytext=(loc, p(loc)-5000))
    plt.show()

def PixelIndices(slicedata, array, col, line='Iron'):
    """NAME: PixelIndices
    PURPOSE: Assigns the value to the element of the array of index locations for a pixel.
    INPUTS:  slicedata      - The array of intensities for a horizontal slice of the sun.
             array          - The array which will hold the index information.
             row            - The row which is being operated upon.
             col            - The column which is being operated upon.
             line           - The absorption line to look at: Calcium or Oxygen
    OUTPUTS: N/A            - Only reassigns the value of the array element.
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    stokesslice=v9s.SingleStokes(datapath, 'I', slicenumber=400)
                    indexmap=np.zeros((len(disk), len(disk)))
                    CL.PixelIndices(stokesslice, indexmap, 400, 700)
                    polyminloc=CL.PolyMinLoc(flux, xs)
            In [7]: indexmap[400][700]
            Out[7]: 61.789999999999999
    """
    flux=v9s.LoadedDataSpectrum(Islicedata=slicedata, slicedata=slicedata, xpos=col)
    if line=='Iron':
        xs=IronCoreRegion(flux)
    elif line=='Oxygen1':
        xs=Oxygen1CoreRegion(flux)
    elif line=='Oxygen2':
        xs=Oxygen2CoreRegion(flux)
    loc=PolyMinLoc(flux, xs)
    array[col]=loc
    
def IndexMaps(datapath, array):
    """NAME: IndexMap 
    PURPOSE: Given a folder of FITS files and an array of full disk intensity information, return an array containing the index of the core of the absorption line.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             array          - The array of full-disk intensity information.
             line           - The absorption line to look at: Calcium or Oxygen
    OUTPUTS: indexmap       - The array containing the indices of the core of the absorption line for each pixel.
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    indexmap=CL.IndexMap(datapath, disk[1020:1028])
            In [7]: indexmap[4][29:35]
            Out[7]: array([  0.   ,   0.   ,  63.016,  60.264,  61.748,  61.472])
    """
    ironmap=np.zeros(len(array))
    oxygen1map=np.zeros(len(array))
    oxygen2map=np.zeros(len(array))
    solarboolean=LF.ContainsSun(datapath=datapath, disk=array)
    gapcols=LF.GapCols(datapath)
    limbs=LF.LimbIndices(datapath, array, solarboolean)
    slicedata=v9s.SingleStokes(datapath=datapath, stokes='I', slicenumber=1024)
    j=0
    while(j<2048):
        PixelIndices(slicedata, ironmap,  j, line='Iron')
        PixelIndices(slicedata, oxygen1map,  j, line='Oxygen1')
        PixelIndices(slicedata, oxygen2map,  j, line='Oxygen2')
        j=j+1
    return ironmap, oxygen1map, oxygen2map

def SaveIndexMaps(datapath, disk, outputtype='.fits', outputpath='local'):
    """NAME: IndexMap 
    PURPOSE: Given a folder of FITS files and an array of full disk intensity information, return an array containing the index of the core of the absorption line.
    INPUTS:  datapath       - The folder name containing all of the FITS files for each slice.
             array          - The array of full-disk intensity information.
             line           - The absorption line to look at: Calcium or Oxygen
    OUTPUTS: indexmap       - The array containing the indices of the core of the absorption line for each pixel.
    EXAMPLE: 
            In [6]: datapath='k4v9s160517t213238_oid114635206372132_cleaned'
                    disk=fits.getdata('/Users/jhamer/Research/Output/k4v9s160517t213238/FullDiskPlots /k4v9s160517t213238_oid114635206372132_cleanedI80fulldisk.fits')
                    indexmap=CL.IndexMap(datapath, disk[1020:1028])
            In [7]: indexmap[4][29:35]
            Out[7]: array([  0.   ,   0.   ,  63.016,  60.264,  61.748,  61.472])
    """        
    ironmap, oxygen1map, oxygen2map=IndexMaps(datapath=datapath, array=disk)
    filen=datapath[0:18]
    if outputpath=='local':
        baseoutput='/Users/jhamer/Research/Output'
        outputpath=baseoutput+'/'+str(filen)+'/IndexMaps'
    else:
        outputpath=outputpath
    if os.path.isdir(baseoutput+'/'+str(filen))==False:
        os.mkdir(baseoutput+'/'+str(filen))
    if os.path.isdir(outputpath)==False:
        os.mkdir(outputpath)
    if outputtype=='.fits':
        hdu=fits.PrimaryHDU(ironmap)
        fn=str(datapath)+'IronIndexMap.fits'
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        hdu.writeto(fullpath)
        hdu=fits.PrimaryHDU(oxygen1map)
        fn=str(datapath)+'Oxygen1IndexMap.fits'
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        hdu.writeto(fullpath)
        hdu=fits.PrimaryHDU(oxygen2map)
        fn=str(datapath)+'oxygen2IndexMap.fits'
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        hdu.writeto(fullpath)
        #v9s.ArraytoFITSImage(array=ironmap, outputpath=outputpath, outputname=str(datapath)+'IronIndexMap' )
        #v9s.ArraytoFITSImage(array=oxygen1map, outputpath=outputpath, outputname=str(datapath)+'Oxygen1IndexMap' )
        #v9s.ArraytoFITSImage(array=oxygen2map, outputpath=outputpath, outputname=str(datapath)+'Oxygen2IndexMap' )
    elif outputtype=='visual':
        plt.clf()
        fig=plt.imshow(ironmap, origin='lower')
        plt.title('Iron Index Map')
        plt.show(fig)
    else:
        fig=plt.imshow(ironmap)
        plt.colorbar()
        fn=str(filen)+'IronIndexMap'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Iron Index Map')
        plt.savefig(fullpath)
        plt.close()
        fig=plt.imshow(oxygen1map)
        plt.colorbar()
        fn=str(filen)+'Oxygen1IndexMap'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Oxygen1 Index Map')
        plt.savefig(fullpath)
        plt.close()
        fig=plt.imshow(oxygen2map)
        plt.colorbar()
        fn=str(filen)+'Oxygen2IndexMap'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Oxygen2 Index Map')
        plt.savefig(fullpath)
        plt.close()