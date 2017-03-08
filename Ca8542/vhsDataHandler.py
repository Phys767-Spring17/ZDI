from astropy.io import fits
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def SliceNumber(num):
    """NAME: SliceNumber 
    PURPOSE: Given a row index, returns the string version of the number with 4 digits.
    INPUTS:  num          - The row number from 0 to 2047.
    OUTPUTS: str(num)     - The corresponding 4 digit string from '0000' to '2047'
    EXAMPLE:
    """
    num=int(num)
    if num<10:
        return "000"+str(num)
    elif num<100:
        return "00"+str(num)  
    elif num<1000:
        return "0"+str(num)   
    else:
        return str(num)
    return num
            
    
def StokesIndex(letter):
    """NAME: StokesIndex 
    PURPOSE: Given an element of the stokes profile, returns a corresponding index.
    INPUTS:  letter       - The element of the stokes profile: I, Q, U, or V
    OUTPUTS: stokesindex  - The corresponding index based on the FITS files: 0, 1, 2, or 3
    EXAMPLE: 
    In [2]: i=StokesIndex('I')
    In [3]: i
    Out[3]: 0
    """
    stokesnames={'I':0, 'Q':1, 'U':2, 'V':3}
    stokesindex=stokesnames[letter]
    return stokesindex

def ArraytoFITSImage(array, outputpath, outputname):
    """NAME: ArraytoFITSImage
    PURPOSE: Saves an array corresponding to pixel values to a FITS file.
    INPUTS:  array        - The  array of pixel values
             outputpath   - The path to write the FITS file to
             outputname   - The name of the FITS file
    OUTPUTS: N/A          - Returns nothing. Only writes the FITS file.
    EXAMPLE:
        In [2]: fulldisk=FullDisk('k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', wavelengthindex=100)
                datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                disk=fulldisk
                datapath='k4vhs160409t174101_oid114602236001741_cleaned'
                outputpath=datapath
                stokes='I'
                wavelengthindex=100
                outputtype='.fits'
                filen=datapath[0:18]
                baseoutput='/Users/jhamer/Research/Output'
                outputpath=baseoutput+'/'+str(filen)
                if os.path.isdir(outputpath)==False:
                    os.mkdir(outputpath)
                if outputtype==fits:
                    ArraytoFITSImage(array=disk, outputpath=outputpath, outputname=str(datapath)+str(stokes)+str(wavelengthindex)+'fulldisk')
    """
    hdu=fits.PrimaryHDU(array)
    fn=str(outputname)+'.fits'
    path=str(outputpath)
    fullpath=os.path.join(path, fn)
    hdu.writeto(fullpath)

def FullDisk(datapath, stokes, wavelengthindex):
    """NAME: FullDisk
    PURPOSE: Returns an array of the pixel values for the full solar disk at specific wavelength index.
    INPUTS:  datapath        - The final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             wavelengthindex - The index of wavelength division from 0 to 127 
    OUTPUTS: rowdata         - The array of rows of pixel values
    EXAMPLE:
        In [2]: fulldisk=FullDisk('k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', wavelengthindex=100)
        In [3]: fulldisk
        Out[3]: [array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), ... ]
    """
    basepath='/Users/jhamer/Research/Data'
    prefix=basepath+'/'+str(datapath)
    filen=datapath[0:18]
    rowdata=[]
    for i in range(2048):
        data=fits.getdata(prefix+'/'+str(filen)+"_"+SliceNumber(i)+".fts.gz")
#        if i<10:
#            data=fits.getdata(prefix+'/'+str(filen)+"_000"+str(i)+".fts.gz")
#        elif i<100:
#            data=fits.getdata(prefix+'/'+str(filen)+"_00"+str(i)+".fts.gz")
#        elif i<1000:
#            data=fits.getdata(prefix+'/'+str(filen)+"_0"+str(i)+".fts.gz")
#        else:
#            data=fits.getdata(prefix+'/'+str(filen)+"_"+str(i)+".fts.gz")
        stokesindex=StokesIndex(stokes)
        row=data[stokesindex][wavelengthindex]
        rowdata.append(row)
    return rowdata

    
def FullDiskPlot(datapath, stokes, wavelengthindex, outputtype='.fits', outputpath='local'):
    """NAME: FullDiskPlot
    PURPOSE: Utilizes FullDisk and ArraytoFITSImage or matplotlib to save the image of the full solar disk at a specific wavelength index for a given element of the stokes vector.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             wavelengthindex - The index of wavelength division from 0 to 127
             outputtype      - The output format: '.fits', '.pdf', '.png', etc. If 'visual' it shows the matplotlib figure.
             outputpath      - The path to write the file to - if the path is not already a directory, it is made. If 'local', it goes to '/Users/jhamer/Research/Output' and makes a directory with the name of the file.
    OUTPUTS: N/A             - Returns nothing. Only saves the image or FITS file.
    EXAMPLE:
        In [2]: FullDiskPlot(datapath='k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', wavelengthindex=100, outputpath='visual')
    """
    disk=FullDisk(datapath=datapath, stokes=stokes, wavelengthindex=wavelengthindex)
    filen=datapath[0:18]
    if outputpath=='local':
        baseoutput='/Users/jhamer/Research/Output'
        outputpath=baseoutput+'/'+str(filen)+'/FullDiskPlots'
    else:
        outputpath=outputpath
    if os.path.isdir(outputpath)==False:
        os.mkdir(outputpath)
    if outputtype=='.fits':
        ArraytoFITSImage(array=disk, outputpath=outputpath, outputname=str(datapath)+str(stokes)+str(wavelengthindex)+'fulldisk' )
    elif outputtype=='visual':
        plt.clf()
        fig=plt.imshow(disk, origin='lower')
        plt.title('Stokes '+str(stokes)+', Wavelength Index: '+str(wavelengthindex))
        plt.show(fig)
    else:
        fig=plt.imshow(disk)
        plt.colorbar()
        fn=str(filen)+str(stokes)+str(wavelengthindex)+'fulldisk'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Stokes '+str(stokes)+', Wavelength Index: '+str(wavelengthindex))
        plt.savefig(fullpath)
        plt.close()

def SingleStokes(datapath, stokes, slicenumber):
    """NAME: SingleStokes
    PURPOSE: Returns an array of the pixel values for an element of the stokes vector for a horizontal slice of the solar disk.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             slicenumber     - The vertical slicenumber from 0 to 2047
    OUTPUTS: stokesslice     - The array of rows of pixel values
    EXAMPLE:
        In [2]: ss=SingleStokes('k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', slicenumber='0500')
        In [3]: ss
        Out[3]: [array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                 array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), ... ]
    """
    basepath='/Users/jhamer/Research/Data'
    prefix=basepath+'/'+str(datapath)
    filen=datapath[0:18]
    data=fits.getdata(prefix+'/'+str(filen)+'_'+SliceNumber(slicenumber)+'.fts.gz')
    stokesindex=StokesIndex(stokes)
    stokesslice=data[stokesindex]
    return stokesslice

def SingleStokesPlot(datapath, stokes, slicenumber, outputtype='.fits', outputpath='local'):
    """NAME: SingleStokesPlot
    PURPOSE: Utilizes SingleStokes and ArraytoFITSImage or matplotlib to save the image of the given stokes vector for wavelength vs xposition on the solar disk.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             slicenumber     - The vertical slicenumber from 0 to 2047
             outputtype      - The output format: '.fits', '.pdf', '.png', etc. If 'visual' it shows the matplotlib figure.
             outputpath      - The path to write the file to - if the path is not already a directory, it is made. If 'local', it goes to '/Users/jhamer/Research/Output' and makes a directory with the name of the file.
    OUTPUTS: N/A             - Returns nothing. Only saves the image or FITS file.
    EXAMPLE:
        In [2]: SingleStokesPlot(datapath='k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', slicenumber=500, outputtype='.fits')
    """
    ss=SingleStokes(datapath=datapath, stokes=stokes, slicenumber=slicenumber)
    filen=datapath[0:18]
    if outputpath=='local':
        baseoutput='/Users/jhamer/Research/Output'
        outputpath=baseoutput+'/'+str(filen)+'/SingleStokesPlots'
    else:
        outputpath=outputpath
    if os.path.isdir(outputpath)==False:
        os.mkdir(outputpath)
    if outputtype=='.fits':
        ArraytoFITSImage(array=ss, outputpath=outputpath, outputname=str(datapath)+str(stokes)+SliceNumber(slicenumber)+'singlestokes' )
    elif outputtype=='visual':
        plt.clf()
        fig=plt.imshow(ss, origin='lower')
        plt.title('Stokes '+str(stokes)+', Slice Number: '+SliceNumber(slicenumber))
        plt.xlabel('x position')
        plt.ylabel('Wavelength')
        plt.colorbar()
        plt.show(fig)
    else:
        fig=plt.imshow(ss)
        plt.colorbar()
        fn=str(filen)+str(stokes)+SliceNumber(slicenumber)+'singlestokes'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Stokes '+str(stokes)+', Slice Number: '+SliceNumber(slicenumber))
        plt.xlabel('x position')
        plt.ylabel('Wavelength')
        plt.colorbar()
        plt.savefig(fullpath)
        plt.close()
        
def Spectrum(datapath, stokes, slicenumber, xpos):
    """NAME: Spectrum
    PURPOSE: Returns an array of flux values over wavelength for a specific location on the solar disk.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             slicenumber     - The vertical slicenumber from 0 to 2047
             xpos            - The x position on the solar disk from 0 to 2047.
    OUTPUTS: flux            - The array of flux values over wavelength.
    EXAMPLE:
        In [2]: flux=Spectrum('k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', slicenumber='0500', xpos=900)
        In [3]: flux
        Out[3]: [..., 0.0, 0.0, 15728.104, 15636.329, 15563.278, 15477.442, 15378.333, ...]
    """    
    basepath='/Users/jhamer/Research/Data'
    prefix=basepath+'/'+str(datapath)
    filen=datapath[0:18]
    data=fits.getdata(prefix+'/'+str(filen)+'_'+SliceNumber(slicenumber)+'.fts.gz')
    stokesindex=StokesIndex(stokes)
    flux=[]
    for i in range(len(data[stokesindex])):
        flux.append(data[stokesindex][i][xpos])
    return flux

def LoadedDataSpectrum(slicedata, xpos):
    """NAME: Spectrum
    PURPOSE: Returns an array of flux values over wavelength for a specific location on the solar disk.
    INPUTS:  slicedata       - The  array of data for a stokes element at a certain slice.
             xpos            - The x position on the solar disk from 0 to 2047.
    OUTPUTS: flux            - The array of flux values over wavelength.
    EXAMPLE:
        In [2]: ss=SingleStokes('k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', slicenumber='0500')
                flux=LoadedDataSpectrum(ss, xpos=900)
        In [3]: flux
        Out[3]: [..., 0.0, 0.0, 15728.104, 15636.329, 15563.278, 15477.442, 15378.333, ...]
    """    
    flux=[]
    for i in range(len(slicedata)):
        flux.append(slicedata[i][xpos])
    return flux

def SpectrumPlot(datapath, stokes, slicenumber, xpos, outputtype='.fits', outputpath='local'):
    """NAME: SpectrumPlot
    PURPOSE: Utilizes Spectrum and ArraytoFITSImage or matplotlib to save the image of the given stokes vector for wavelength vs xposition on the solar disk.
    INPUTS:  datapath        - The  final directory in which the FITS files are located
             stokes          - The element of the stokes profile: I, Q, U, or V
             slicenumber     - The vertical slicenumber from 0 to 2047 
             xpos            - The x position on the solar disk from 0 to 2047.
             outputtype      - The output format: '.fits', '.pdf', '.png', etc. If 'visual' it shows the matplotlib figure.
             outputpath      - The path to write the file to - if the path is not already a directory, it is made. If 'local', it goes to '/Users/jhamer/Research/Output' and makes a directory with the name of the file.
    OUTPUTS: N/A             - Returns nothing. Only saves the image or FITS file.
    EXAMPLE:
        In [2]: SpectrumPlot(datapath='k4vhs160409t174101_oid114602236001741_cleaned', stokes='I', slicenumber=0500, xpos=900, outputtype='.pdf')
    """
    flux=Spectrum(datapath=datapath, stokes=stokes, slicenumber=slicenumber, xpos=xpos)
    filen=datapath[0:18]
    if outputpath=='local':
        baseoutput='/Users/jhamer/Research/Output'
        outputpath=baseoutput+'/'+str(filen)+'/Spectra'
    else:
        outputpath=outputpath
    if os.path.isdir(outputpath)==False:
        os.mkdir(outputpath)
    if outputtype=='.fits':
        ArraytoFITSImage(array=flux, outputpath=outputpath, outputname=str(datapath)+str(stokes)+SliceNumber(slicenumber)+'xpos'+str(xpos)+'spectrum' )
    elif outputtype=='visual':
        plt.clf()
        fig=plt.plot(flux)
        plt.title('Stokes '+str(stokes)+', Slice Number: '+SliceNumber(slicenumber)+', x Position: '+str(xpos))
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.show(fig)
    else:
        plt.clf()
        fig=plt.plot(flux)
        fn=str(filen)+str(stokes)+str(slicenumber)+'xpos'+str(xpos)+'spectrum'+str(outputtype)
        path=str(outputpath)
        fullpath=os.path.join(path, fn)
        plt.title('Stokes '+str(stokes)+', Slice Number: '+SliceNumber(slicenumber)+', x Position: '+str(xpos))
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.savefig(fullpath)
        plt.close()