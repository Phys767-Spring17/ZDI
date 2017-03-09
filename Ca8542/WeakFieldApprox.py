import vhsDataHandler as vhs
import CoreLocation as CL
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as sym
from astropy.io import fits
import scipy.special as sps
import scipy.signal as ss
from datetime import datetime

def LambdaNaught(calciummap, watermap):
    """NAME: LambdaNaught 
    PURPOSE: Returns an array for the wavelength of the calcium absorption line core for the full disk.
    INPUTS:  calciummap      - The array of indices of the calcium line core.
             watermap        - The array of indices of the water line core
    OUTPUTS: lambdanaughts   - The array of wavelengths of the calcium absorption line core
    EXAMPLE: 
            In [6]: from astropy.io import fits
                    ca=fits.getdata('CalciumAbsorptionCoreIndex.fits')
                    wa=fits.getdata('WaterAbsorptionCoreIndex.fits')
                    lns=LambdaNaught(ca, wa)
            In [7]: lns
            Out[7]: [array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), ... ]
    """
    lambdanaughts=8540.7959*np.ones((len(calciummap), len(calciummap)))
    distance=calciummap-watermap
    wavelengthdistance=distance*36.5*10**-3
    lambdanaughts=lambdanaughts+wavelengthdistance
    return lambdanaughts

def Alpha(lambdanaught, row, col):
    """NAME: Alpha 
    PURPOSE: Returns constant alpha used in calculation of C. alpha=4.66*10^-13*geff*lambda_naught^2
    INPUTS:  lambdanaughts   - The array of wavelengths of the calcium absorption line core
             row             - The row of the full disk image, from 0 to 2047
             col             - The row of the full disk image, from 0 to 2047
    OUTPUTS: alpha           - The constant alpha used in calculation of C
    EXAMPLE: 
            In [6]: from astropy.io import fits
                    ca=fits.getdata('CalciumAbsorptionCoreIndex.fits')
                    wa=fits.getdata('WaterAbsorptionCoreIndex.fits')
                    lns=LambdaNaught(ca, wa)
                    row=1500
                    col=378
                    alpha=Alpha(lns, row, col)
            In [7]: alpha
            Out[7]: 0.005
    """
    geff=1.10
    alpha=-4.6686*10**-13*geff*(lambdanaught[row][col]**2)
    return alpha

def A1A2A3(Islicedata, Vslicedata, col):
    """NAME: A1A2A3 
    PURPOSE: Returns A1, A2, A3, N used in calculation of C and marginal pdfs
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Vslicedata   - The array of information for stokes V at a horizontal slice (is divided by I)
             col          - The col of the full disk image, from 0 to 2047
    OUTPUTS: A1           - A1 is the sum of squared stokes V values in calcium core region
             A2           - A2 is the sum of stokes V * derivative of I in calcium core region
             A3           - A3 is the sum derivative of I squared in calcium core region
             N            - N is the number of points around the core region included
    EXAMPLE: 
            In [6]: ssI=vhs.SingleStokes('k4vhs160409t174101_oid114602236001741_cleaned', 'I', row)
                    ssV=vhs.SingleStokes('k4vhs160409t174101_oid114602236001741_cleaned', 'V', row)
                    col=378
                    A1, A2, A3, N=A1A2A3(ssI, ssV, col)
            In [7]: print(A1, A2, A3, N)
            Out[7]: (0.000940422, -0.0061357394, 0.83149779, 12)
    """
    Iflux=np.array(vhs.LoadedDataSpectrum(Islicedata, col))
    Vflux=np.array(vhs.LoadedDataSpectrum(Vslicedata, col))
    Vflux=Vflux*Iflux
    Vflux=Vflux/np.max(Iflux[30:100])
    Iflux=Iflux/np.max(Iflux[30:100])
    core=np.arange(55,67, 1)
    Icore=Iflux[core]
    Vcore=Vflux[core]
    gradI=np.gradient(Icore)/(36.5*10**-3)
    A1=np.ndarray.sum(Vcore**2)
    A2=np.ndarray.sum(Vcore*gradI)
    A3=np.ndarray.sum(gradI**2)
    N=len(core)
    return A1, A2, A3, N

#def C(A1, A2, A3, alpha):
    #f, B=sym.symbols('f B')
#    C=A1-alpha*f*B*A2+(alpha*f*B)**2*A3
#    return C

def StandardRectangleFunction(f):
    """NAME: StandardRectangleFunction 
    PURPOSE: Prior distribution for fill factor, f.
    INPUTS:  f   - The value of the filling factor, from 0 to 1.
    OUTPUTS: 0, 1, or .5 depending on value of f
    EXAMPLE: 
            In [6]: srf=StandardRectangleFunction(0.5)
            In [7]: srf
            Out[7]: 1
    """
    arg=f-0.5
    if np.abs(arg)> 0.5:
        return 0
    elif np.abs(arg)< 0.5:
        return 1
    elif np.abs(arg)==0.5:
        return 0.5

def GammaDiff(C, N, sigmanmin=10**-5, sigmanmax=10**-2):
    """NAME: GammaDiff 
    PURPOSE: Returns the value of the differences of the two incomplete gamma functions in the generalized posterior distribution
    INPUTS:  C               - The value of constant C
             N               - N is the number of points around the core region included
             sigmanmin       - The min uncertainty on observations
             sigmanmax       - The max uncertainty on observations
    OUTPUTS: minterm-maxterm - The value between 0 and 1 
    EXAMPLE: See function fintegrand below for usage.
    """
    minterm=sps.gammainc(N/2, C/(2*sigmanmin**2))
    maxterm=sps.gammainc(N/2, C/(2*sigmanmax**2))
    return minterm-maxterm

def ErfDiff(B, sigmamax=3000, sigmamin=1):
    """NAME: ErfDiff 
    PURPOSE: Returns the value of the differences of the two error functions in the generalized posterior distribution
    INPUTS:  B               - The value of line of sight magnetic field
             N               - N is the number of points around the core region included
             sigmamin        - The min uncertainty on LOS B field
             sigmamax        - The max uncertainty on LOS B field
    OUTPUTS: minterm-maxterm - The value between 0 and 1 
    EXAMPLE: See function fintegrand below for usage.
    """
    minterm=sps.erf(B/(np.sqrt(2)*sigmamin))
    maxterm=sps.erf(B/(np.sqrt(2)*sigmamax))
    return minterm-maxterm

#def GeneralizedPosterior(f, N, C, B, sigmanmin=10**-5, sigmanmax=10**-2, sigmamax=3000, sigmamin=1):
#    srf=StandardRectangleFunction(f=f)
#    gd=GammaDiff(C=C, N=N, sigmanmin=sigmanmin, sigmanmax=sigmanmax)
#    ed=ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)
#    post=.5*(2*np.pi)**(-N/2)*C**(-N/2)*(1/np.abs(B))*srf*gd*ed
#    return post

def fintegrand(B, f, A1, A2, A3, N, alpha, sigmanmin=10**-5, sigmanmax=10**-2, sigmamax=3000, sigmamin=1):
    """NAME: fintegrand 
    PURPOSE: Returns the integrand for numerical integration to find the posterior distribution of f
    INPUTS:  B               - The value of line of sight magnetic field
             f               - The value of the filling factor, from 0 to 1.
             A1              - A1 is the sum of squared stokes V values in calcium core region
             A2              - A2 is the sum of stokes V * derivative of I in calcium core region
             A3              - A3 is the sum derivative of I squared in calcium core region
             N               - N is the number of points around the core region included
             sigmanmin       - The min uncertainty on observations
             sigmanmax       - The max uncertainty on observations
             sigmamin        - The min uncertainty on LOS B field
             sigmamax        - The max uncertainty on LOS B field
    OUTPUTS: integrand       - To be integrated numerically in fpixelmarginal
    EXAMPLE: See function fpixelmarginal below for usage.
    """
    N=N
    C=A1-alpha*f*B*A2+(alpha*f*B)**2*A3
    return .5*(2*np.pi)**(-N/2)*C**(-N/2)*(1/np.abs(B))*StandardRectangleFunction(f)*GammaDiff(C=C, N=N, sigmanmin=sigmanmin, sigmanmax=sigmanmax)*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def Bintegrand(f, B, A1, A2, A3, N, alpha, sigmanmin=10**-5, sigmanmax=10**-2, sigmamax=3000, sigmamin=1):
    """NAME: Bintegrand 
    PURPOSE: Returns the integrand for numerical integration to find the posterior distribution of B
    INPUTS:  f               - The value of the filling factor, from 0 to 1.
             B               - The value of line of sight magnetic field
             A1              - A1 is the sum of squared stokes V values in calcium core region
             A2              - A2 is the sum of stokes V * derivative of I in calcium core region
             A3              - A3 is the sum derivative of I squared in calcium core region
             N               - N is the number of points around the core region included
             sigmanmin       - The min uncertainty on observations
             sigmanmax       - The max uncertainty on observations
             sigmamin        - The min uncertainty on LOS B field
             sigmamax        - The max uncertainty on LOS B field
    OUTPUTS: integrand       - To be integrated numerically in Bpixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    """
    N=N
    C=A1-alpha*f*B*A2+(alpha*f*B)**2*A3
    return .5*(2*np.pi)**(-N/2)*C**(-N/2)*(1/np.abs(B))*StandardRectangleFunction(f)*GammaDiff(C=C, N=N, sigmanmin=sigmanmin, sigmanmax=sigmanmax)*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def fpixelmarginal(lambdanaughts, Islicedata, Vslicedata, row, col, integration='quad'):
    """NAME: fpixelmarginal 
    PURPOSE: Returns the pdf of f for a pixel
    INPUTS:  lambdanaughts  - The array of wavelengths of the calcium absorption line core
             Islicedata     - The array of information for stokes I at a horizontal slice
             Vslicedata     - The array of information for stokes V at a horizontal slice (is divided by I)
             row            - The row of the full disk image, from 0 to 2047
             col            - The col of the full disk image, from 0 to 2047
             integration    - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: fprobs         - The probability values for the 50 values of fvalues=np.linspace(0, 1, 50)
    EXAMPLE: See function fMAPmap below for usage.
    """    
    fvalues=np.linspace(0, 1, 50)
    fprobs=[]
    alpha=Alpha(lambdanaughts, row, col)
    A1, A2, A3, N=A1A2A3(Islicedata, Vslicedata, col)
    for i in range(len(fvalues)):
        if integration=='quad':
            fprobs.append(sp.integrate.quad(fintegrand, 1, 2000, args=(fvalues[i], A1, A2, A3, N, alpha))[0])
        if integration=='romberg':
            fprobs.append(sp.integrate.romberg(fintegrand, 1, 2000, args=(fvalues[i], A1, A2, A3, N, alpha)))
    return fprobs

def ffullmarginal(datapath, calciummap, watermap):
    """NAME: ffullmarginal 
    PURPOSE: Returns the pdf for f for each pixel in the disk of the sun
    INPUTS:  datapath        - The final directory in which the FITS files are located
             calciummap      - The array of indices of the calcium line core.
             watermap        - The array of indices of the water line core
    OUTPUTS: marginalmap     - An array for pdfs for f for each pixel
    EXAMPLE: 
    """    
    lns=LambdaNaught(calciummap, watermap)
    marginalmap=[]
    for i in range(2048):
        row=[]
        ssI=vhs.SingleStokes(datapath, 'I', i)
        ssV=vhs.SingleStokes(datapath, 'V', i)
        for j in range(2048):
            if lns[i][j]==0:
                row.append(np.zeros(100))
            else:
                row.append(fpixelmarginal(lns, ssI, ssV, i, j))
        marginalmap.append(row)
    return marginalmap

def fMAPmap(datapath, calciummap, watermap, integration):
    """NAME: fMAPmap 
    PURPOSE: Returns the MAP solution for f for each pixel in the sun. 
    INPUTS:  datapath        - The final directory in which the FITS files are located
             calciummap      - The array of indices of the calcium line core.
             watermap        - The array of indices of the water line core.
             integration     - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: marginalmap     - An array of the MAP solution for f for each pixel
    EXAMPLE: 
    """    
    lns=LambdaNaught(calciummap, watermap)
    marginalmap=[]
    for i in np.arange(1470,1510,1):
        print(i, str(datetime.now()))
        row=[]
        ssI=vhs.SingleStokes(datapath, 'I', i)
        ssV=vhs.SingleStokes(datapath, 'V', i)
        for j in np.arange(370,410,1):
            if lns[i][j]==0:
                row.append(0)
            else:
                pdf=fpixelmarginal(lns, ssI, ssV, i, j, integration)
                MAPloc=np.argmax(pdf)
                MAP=np.linspace(0,1,50)[MAPloc]
                row.append(MAP)
        marginalmap.append(row)
    return marginalmap            
    
def Bpixelmarginal(lambdanaughts, Islicedata, Vslicedata, row, col, integration='quad'):
    """NAME: Bpixelmarginal 
    PURPOSE: Returns the pdf of B for a pixel
    INPUTS:  lambdanaughts  - The array of wavelengths of the calcium absorption line core
             Islicedata     - The array of information for stokes I at a horizontal slice
             Vslicedata     - The array of information for stokes V at a horizontal slice (is divided by I)
             row            - The row of the full disk image, from 0 to 2047
             col            - The col of the full disk image, from 0 to 2047
             integration    - Numerical integration method: 'quad' or 'romberg'. Romberg is very different for B and takes ~10 times as long for B.
    OUTPUTS: Bprobs         - The probability values for the 50 values of Bvalues=np.logspace(0, 3.3, 50)
    EXAMPLE: See function BMAPmap below for usage.
    """    
    Bvalues=np.logspace(0, 3.3, 50)
    Bprobs=[]
    alpha=Alpha(lambdanaughts, row, col)
    A1, A2, A3, N=A1A2A3(Islicedata, Vslicedata, col)
    for i in range(len(Bvalues)):
        if integration=='quad':
            Bprobs.append(sp.integrate.quad(Bintegrand, 0, 1, args=(Bvalues[i], A1, A2, A3, N, alpha))[0])
        if integration=='romberg':
            Bprobs.append(sp.integrate.romberg(Bintegrand, 0, 1, args=(Bvalues[i], A1, A2, A3, N, alpha)))
    return Bprobs

def Bfullmarginal(datapath, calciummap, watermap):
    """NAME: Bfullmarginal 
    PURPOSE: Returns the pdf for B for each pixel in the disk of the sun
    INPUTS:  datapath        - The final directory in which the FITS files are located
             calciummap      - The array of indices of the calcium line core.
             watermap        - The array of indices of the water line core
    OUTPUTS: marginalmap     - An array for pdfs for B for each pixel
    EXAMPLE: 
    """   
    lns=LambdaNaught(calciummap, watermap)
    marginalmap=[]
    for i in range(2048):
        row=[]
        ssI=vhs.SingleStokes(datapath, 'I', i)
        ssV=vhs.SingleStokes(datapath, 'V', i)
        for j in range(2048):
            if lns[i][j]==0:
                row.append(np.zeros(100))
            else:
                row.append(Bpixelmarginal(lns, ssI, ssV, i, j))
        marginalmap.append(row)
    return marginalmap

def BMAPmap(datapath, calciummap, watermap, integration):
    """NAME: BMAPmap 
    PURPOSE: Returns the MAP solution for B for each pixel in the sun. 
    INPUTS:  datapath        - The final directory in which the FITS files are located
             calciummap      - The array of indices of the calcium line core.
             watermap        - The array of indices of the water line core.
             integration    - Numerical integration method: 'quad' or 'romberg'. Romberg is very different for B and takes ~10 times as long for B.
    OUTPUTS: marginalmap     - An array of the MAP solution for B for each pixel
    EXAMPLE: 
    """    
    lns=LambdaNaught(calciummap, watermap)
    marginalmap=[]
    for i in np.arange(1470,1510,1):
        print(i, str(datetime.now()))
        row=[]
        ssI=vhs.SingleStokes(datapath, 'I', i)
        ssV=vhs.SingleStokes(datapath, 'V', i)
        for j in np.arange(370, 410,1):
            if lns[i][j]==0:
                row.append(0)
            else:
                pdf=Bpixelmarginal(lns, ssI, ssV, i, j, integration)
                MAPloc=np.argmax(pdf)
                MAP=np.logspace(0,3.3,50)[MAPloc]
                row.append(MAP)
        marginalmap.append(row)
    return marginalmap      