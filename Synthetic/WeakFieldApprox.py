import v9sDataHandler as v9s
import numpy as np
import scipy as sp
import scipy.special as sps
from datetime import datetime

def Dispersion(oxygen1map, oxygen2map):
    """NAME: Dispersion 
    PURPOSE: Returns the number of angstroms per wavelength pixel.
    INPUTS:  oxygen1map        - The array of indices of the oxygen line core at 6302.0005 A
             oxygen2map        - The array of indices of the oxygen line core at 6302.0005+0.7622 A
    OUTPUTS: dispersion       - The array of Angstroms/wavelength pixel for each pixel on the image of the Sun
    EXAMPLE: See LambdaNaught below
    """
    distance=oxygen2map-oxygen1map #number of indices between the two terrestrial oxygen absorption lines
    dispersion=0.76220703/distance #the lines are 0.76220703 Angstroms apart
    return dispersion #the Angstroms/wavelength pixel

def LambdaNaught(ironmap, oxygen1map, dispersion):
    """NAME: LambdaNaught 
    PURPOSE: Returns an array for the wavelength of the calcium absorption line core for the full disk.
    INPUTS:  ironmap       - The array of indices of the iron line core.
             oxygen1map     - The array of indices of the first oxygen line core
             dispersion    - The array of Angstroms/wavelength pixel for each pixel on the image of the Sun
    OUTPUTS: lambdanaughts - The array of wavelengths of the iron absorption line core
    EXAMPLE: 
            In [6]: from astropy.io import fits
                    fe=fits.getdata('IronAbsorptionCoreIndex.fits')
                    ox1=fits.getdata('Oxygen1AbsorptionCoreIndex.fits')
                    ox2=fits.getdata('Oxygen2AbsorptionCoreIndex.fits')
                    dispersion=Dispersion(ox1, ox2)
                    lns=LambdaNaught(fe, ox1, dispersion)
            In [7]: lns
            Out[7]: [array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), ... ]
    """
    lambdanaughts=6302.0005*np.ones(len(ironmap)) #base of 6302.0005A for every pixel #for real data change to np.ones((len(ironmap), len(ironmap)))
    distance=ironmap-oxygen1map #number of indices between Fe6302 Line core and the first terrestrial oxygen line
    wavelengthdistance=distance*dispersion #converts index distance to distance in Angstroms
    lambdanaughts=lambdanaughts+wavelengthdistance #adds the distance to the base 6302.0005 location of the first oxygen line
    return lambdanaughts #array of locations of iron line core in angstroms
    
def VminusdIvar(Islicedata, Vslicedata, dispersion, alpha, f, B, col):
    """NAME: VminusdIvar 
    PURPOSE: Returns noise of data
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Vslicedata   - The array of information for stokes V at a horizontal slice (is divided by I)
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             f            - The value of the filling factor
             B            - The value of the LOS B field    n iron core region
             col          - The col of the full disk image, from 0 to 2047
    OUTPUTS: sigma        - The noise to be plugged into the Generalized Posterior
    EXAMPLE: 
    """
    Iflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Islicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Vflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Vslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel
    Vcontinuum=Vflux[97:115] #picks out the wavelengths at which there is continuum
    Icontinuum=Iflux[97:115] #picks out the wavelengths at which there is continuum
    deriv=np.gradient(Icontinuum)/dispersion #the wavelength derivative of Intensity, calculated by chain rule: dI/dpixel * dpixel/dAngstrom
    dIvar=np.var(deriv) #variance of the derivative
    Vvar=np.var(Vcontinuum) #variance in V
    cov=np.cov((Vcontinuum,deriv))[0][1] #the covariance is a matrix, the terms on the opposite diagonal are the cross terms
    #var=Vvar+(alpha*f*B)**2.*dIvar-2.*(alpha*f*B)*cov #based on Var(X+Y)=Var(X)+Var(Y)-2*Cov(X,Y) - for Var(dI) the constants come out twice
    var=Vvar #simplest case
    sigma=np.sqrt(var) #noise/st. dev. is sqrt(variance)
    return sigma #in Asensios Sec. 3.1 on the left side middle paragraph: "variance of terms inside the parentheses"
    
def Alpha(lambdanaught):
    """NAME: Alpha 
    PURPOSE: Returns array of constant alpha for each pixel: alpha=4.66*10^-13*geff*lambda_naught^2
    INPUTS:  lambdanaught   - The array of wavelengths of the calcium absorption line core
    OUTPUTS: alpha           - The constant alpha: alpha=4.66*10^-13*geff*lambda_naught^2
    EXAMPLE: 
            In [6]: lns=LambdaNaught(fe, ox1, disp)
                    alpha=Alpha(lns)
            In [7]: alpha
            Out[7]: [array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
                     array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), ... ]
    """
    geff=2.5 #g-Lande factor: 2.5 for Fe6302, 1.1 for Ca8542
    alpha=-4.6686*10**-13*geff*(lambdanaught**2.) #defined on bottom of page 2 of Asensios Ramos 2011
    return alpha

def A1A2A3(Islicedata, Vslicedata, f, B, col, dispersion, alpha, ironloc):
    """NAME: A1A2A3 
    PURPOSE: Returns A1, A2, A3, N defined on page 3 of Asensios
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Vslicedata   - The array of information for stokes V at a horizontal slice (is divided by I)
             col          - The col of the full disk image, from 0 to 2047
             f            - The value of the filling factor
             B            - The value of the LOS B field
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             ironloc      - Index of iron absorption core         
    OUTPUTS: A1           - A1 is the sum of squared stokes V values in iron core region
             A2           - A2 is the sum of stokes V * derivative of I in iron core region
             A3           - A3 is the sum derivative of I squared in iron core region
             N            - N is the number of wavelength pixels around the core region included
    EXAMPLE: 
    """
    Iflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Islicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Vflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Vslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel    
    ironloc=np.int(ironloc) #the iron absorption core loc for the single pixel
    core=np.arange(ironloc-2, ironloc+3,1) #the indices of the region of the absorption line in the spectrum
    Icore=Iflux[core] #selects out the region of iron absorption in the spectrum
    Vcore=Vflux[core] #selects out the region of iron absorption in the spectrum
    gradI=np.gradient(Icore)/dispersion #dI/dlambda
    sigma=VminusdIvar(Islicedata=Islicedata,Vslicedata=Vslicedata,dispersion=dispersion,alpha=alpha,f=f,B=B,col=col) #calculate noise value
    A1=np.ndarray.sum(Vcore**2.)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos
    A2=np.ndarray.sum(Vcore*gradI)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    A3=np.ndarray.sum(gradI**2.)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    N=len(core) #number of wavelength pixels included in summation
    return A1, A2, A3, N
    
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
        return 0.
    elif np.abs(arg)< 0.5:
        return 1.
    elif np.abs(arg)==0.5:
        return 0.5 #http://mathworld.wolfram.com/RectangleFunction.html

def ErfDiff(B, sigmamin=1., sigmamax=3000.):
    """NAME: ErfDiff 
    PURPOSE: Returns the value of the differences of the two error functions in the generalized posterior distribution
    INPUTS:  B               - The value of line of sight magnetic field
             sigmamin        - The min uncertainty on LOS B field
             sigmamax        - The max uncertainty on LOS B field
    OUTPUTS: minterm-maxterm - The value of the erf difference
    EXAMPLE: See function fintegrand below for usage.
    """
    minterm=sps.erf(B/(np.sqrt(2.)*sigmamin))
    maxterm=sps.erf(B/(np.sqrt(2.)*sigmamax))
    return minterm-maxterm

def fintegrand(B, f, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: fintegrand 
    PURPOSE: Returns the integrand for numerical integration to find the posterior distribution of f
    INPUTS:  B               - The value of line of sight magnetic field
             f               - The value of the filling factor, from 0 to 1.
             Islicedata      - The array of information for stokes I at a horizontal slice
             Vslicedata      - The array of information for stokes V at a horizontal slice (is divided by I)
             col             - The col of the full disk image, from 0 to 2047
             dispersion      - The number of angstroms/wavelength pixel
             alpha           - Constant alpha returned by function Alpha above  
             ironloc         - Wavelength of iron absorption core
             sigmamin        - The min possible noise value on B LOS
             sigmamax        - The max possible noise value on B LOS
    OUTPUTS: integrand       - The generalized posterior for f, B_LOS given known noise variance. To be integrated numerically in fpixelmarginal.
    EXAMPLE: See function fpixelmarginal below for usage.
    """
    A1, A2, A3, N=A1A2A3(Islicedata=Islicedata, Vslicedata=Vslicedata, f=f, B=B, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc)
    sigma_d=VminusdIvar(Islicedata=Islicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, f=f, B=B, col=col)
    exp=np.exp(-(A1-(2.*alpha*f*B*A2)+((alpha*f*B)**2.)*A3))
    return .5*(2.*np.pi)**(-N/2.)*sigma_d**(-N)*(1./B)*StandardRectangleFunction(f=f)*exp*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def Bintegrand(f, B, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: Bintegrand 
    PURPOSE: Returns the integrand for numerical integration to find the posterior distribution of B
    INPUTS:  B               - The value of line of sight magnetic field
             f               - The value of the filling factor, from 0 to 1.
             Islicedata      - The array of information for stokes I at a horizontal slice
             Vslicedata      - The array of information for stokes V at a horizontal slice (is divided by I)
             col             - The col of the full disk image, from 0 to 2047
             dispersion      - The number of angstroms/wavelength pixel
             alpha           - Constant alpha returned by function Alpha above  
             ironloc         - Wavelength of iron absorption core
             sigmamin        - The min possible noise value on B LOS
             sigmamax        - The max possible noise value on B LOS
    OUTPUTS: integrand       - The generalized posterior for f, B_LOS given known noise variance. To be integrated numerically in fpixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    """
    A1, A2, A3, N=A1A2A3(Islicedata=Islicedata, Vslicedata=Vslicedata, f=f, B=B, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc)
    sigma_d=VminusdIvar(Islicedata=Islicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, f=f, B=B, col=col)
    exp=np.exp(-(A1-(2.*alpha*f*B*A2)+((alpha*f*B)**2.)*A3))
    return .5*(2.*np.pi)**(-N/2.)*sigma_d**(-N)*(1./B)*StandardRectangleFunction(f=f)*exp*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def JointPosterior(f, B, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: Bintegrand 
    PURPOSE: Returns the integrand for numerical integration to find the posterior distribution of B
    INPUTS:  B               - The value of line of sight magnetic field
             f               - The value of the filling factor, from 0 to 1.
             Islicedata      - The array of information for stokes I at a horizontal slice
             Vslicedata      - The array of information for stokes V at a horizontal slice (is divided by I)
             col             - The col of the full disk image, from 0 to 2047
             dispersion      - The number of angstroms/wavelength pixel
             alpha           - Constant alpha returned by function Alpha above  
             ironloc         - Wavelength of iron absorption core
             sigmamin        - The min possible noise value on B LOS
             sigmamax        - The max possible noise value on B LOS
    OUTPUTS: integrand       - The generalized posterior for f, B_LOS given known noise variance. To be integrated numerically in fpixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    """
    A1, A2, A3, N=A1A2A3(Islicedata=Islicedata, Vslicedata=Vslicedata, f=f, B=B, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc)
    sigma_d=VminusdIvar(Islicedata=Islicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, f=f, B=B, col=col)
    exp=np.exp(-(A1-(2.*alpha*f*B*A2)+((alpha*f*B)**2.)*A3))
    return .5*(2.*np.pi)**(-N/2.)*sigma_d**(-N)*(1./B)*StandardRectangleFunction(f=f)*exp*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def PosteriorMatrix(nf, nB, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    fvals=np.linspace(0, 1, nf)
    Bvals=np.linspace(-2000, 2000, nB)+.00001
    posteriors=np.zeros((nf, nB))
    for i in range(nf):
        for j in range(nB):
            posteriors[i][j]=JointPosterior(f=fvals[i], B=Bvals[j], Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc, sigmamax=sigmamax, sigmamin=sigmamin)
    if np.max(posteriors)>0:
        posteriors=posteriors/np.max(posteriors)
    return posteriors

def pixelmarginals(nf, nB, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1., integration='quad'):
    posteriormatrix=PosteriorMatrix(nf=nf, nB=nB ,Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc, sigmamax=3000., sigmamin=1.)
    fvalues=np.linspace(0,1,nf)
    Bvalues=np.linspace(-2000, 2000, nB)
    fmarginals=sp.integrate.simps(posteriormatrix, Bvalues)
    fmarginals=fmarginals/np.max(fmarginals)
    Bmarginals=sp.integrate.simps(posteriormatrix.T, fvalues)
    Bmarginals=Bmarginals/np.max(Bmarginals)
    return (fvalues, fmarginals), (Bvalues, Bmarginals)
    
def pixelMAPs(nf, nB, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    posteriormatrix=PosteriorMatrix(nf=nf, nB=nB,Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc, sigmamax=3000., sigmamin=1.)
    fvalues=np.linspace(0,1,nf)
    Bvalues=np.linspace(-2000, 2000, nB)
    fmarginals=sp.integrate.simps(posteriormatrix, Bvalues)
    fMAPloc=np.argmax(fmarginals)
    fMAP=fvalues[fMAPloc]
    Bmarginals=sp.integrate.simps(posteriormatrix.T, fvalues)
    BMAPloc=np.argmax(Bmarginals)
    BMAP=Bvalues[BMAPloc]
    return fMAP, BMAP
    
def fpixelmarginal(Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1., integration='quad'):
    """NAME: fpixelmarginal 
    PURPOSE: Returns the pdf of f for a pixel
    INPUTS:  Islicedata      - The array of information for stokes I at a horizontal slice
             Vslicedata      - The array of information for stokes V at a horizontal slice (is divided by I)
             col             - The col of the full disk image, from 0 to 2047
             dispersion      - The number of angstroms/wavelength pixel
             alpha           - Constant alpha returned by function Alpha above  
             ironloc         - Wavelength of iron absorption core
             sigmamin        - The min possible noise value on B LOS
             sigmamax        - The max possible noise value on B LOS
             integration     - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: fprobs          - The probability values for the 50 values of fvalues=np.linspace(0, 1, 50)
    EXAMPLE: See function fMAPmap below for usage.
    """    
    fvalues=np.linspace(0, 1, 100) #the values of the filling factor to test the probability of
    fprobs=[]
    for i in range(len(fvalues)):
        if integration=='quad':
            int1=sp.integrate.quad(func=fintegrand, a=-2000., b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc), points=0)[0]
            #int2=sp.integrate.quad(func=fintegrand, a=0.01, b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))[0]
            #inte=int1+int2
            fprobs.append(int1)
        if integration=='romberg':
            int1=sp.integrate.romberg(function=fintegrand, a=-2000., b=-0.01, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))
            int2=sp.integrate.romberg(function=fintegrand, a=0.01, b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))
            inte=int1+int2
            fprobs.append(inte)
    return fprobs

def fpixelMAP(Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1., integration='quad'):
    """NAME: fpixelmarginal 
    PURPOSE: Returns the pdf of f for a pixel
    INPUTS:  Islicedata      - The array of information for stokes I at a horizontal slice
             Vslicedata      - The array of information for stokes V at a horizontal slice (is divided by I)
             col             - The col of the full disk image, from 0 to 2047
             dispersion      - The number of angstroms/wavelength pixel
             alpha           - Constant alpha returned by function Alpha above  
             ironloc         - Wavelength of iron absorption core
             sigmamin        - The min possible noise value on B LOS
             sigmamax        - The max possible noise value on B LOS
             integration     - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: fprobs          - The probability values for the 50 values of fvalues=np.linspace(0, 1, 50)
    EXAMPLE: See function fMAPmap below for usage.
    """    
    fvalues=np.linspace(0, 1, 100) #the values of the filling factor to test the probability of
    fprobs=[]
    for i in range(len(fvalues)):
        if integration=='quad':
            int1=sp.integrate.quad(func=fintegrand, a=-2000., b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc), points=0)[0]
            #int2=sp.integrate.quad(func=fintegrand, a=0.01, b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))[0]
            #inte=int1+int2
            fprobs.append(int1)
            #inside the args argument of integration, you cannot put the keywords for your arguments, so they MUST be in the right order
        if integration=='romberg':
            int1=sp.integrate.romberg(function=fintegrand, a=-2000., b=-0.01, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))
            int2=sp.integrate.romberg(function=fintegrand, a=0.01, b=2000, args=(fvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))
            inte=int1+int2
            fprobs.append(inte)
    argmax=np.argmax(fprobs)
    MAP=fvalues[argmax]
    return MAP

def ffullmarginal(datapath, ironmap, oxygen1map,oxygen2map):
    """NAME: ffullmarginal 
    PURPOSE: Returns the pdf for f for each pixel in the disk of the sun
    INPUTS:  datapath        - The final directory in which the FITS files are located
             ironmap         - The array of indices of the iron line core.
             oxygen1map       - The array of indices of the first oxygen line core
             oxygen2map       - The array of indices of the second oxygen line core
    OUTPUTS: marginalmap     - An array for pdfs for f for each pixel
    EXAMPLE: 
    """    
    dispersion=Dispersion(oxygen1map=oxygen1map, oxygen2map=oxygen2map) #angstroms/wavelength pixel
    lambdanaughts=LambdaNaught(ironmap=ironmap, oxygen1map=oxygen1map,dispersion=dispersion) #wavelenth of iron line core
    alphas=Alpha(lambdanaught=lambdanaughts) #alpha values
    marginalmap=[]
    for i in range(2048):
        row=[]
        ssI=v9s.SingleStokes(datapath=datapath, stokes='I', row=i) #row stokes I data
        ssV=v9s.SingleStokes(datapath=datapath, stokes='V', row=i) #row stokes V data
        for j in range(2048):
            if lambdanaughts[i][j]==0:
                pdf=np.zeros(100)
                pdf[0]=1 #if the pixel is not on the sun, want the most likely value of f to be zero
                row.append(pdf)
            else:
                row.append(fpixelmarginal(Islicedata=ssI, Vslicedata=ssV, col=j, dispersion=dispersion[i][j], alpha=alphas[i][j], ironloc=ironmap[i][j], sigmamax=3000., sigmamin=1., integration='quad'))
        marginalmap.append(row)
    return marginalmap

def fMAPmap(datapath, ironmap, oxygen1map,oxygen2map):
    """NAME: fMAPmap 
    PURPOSE: Returns the MAP solution for f for each pixel in the sun. 
    INPUTS:  datapath        - The final directory in which the FITS files are located
             ironmap       - The array of indices of the iron line core.
             oxygen1map     - The array of indices of the first oxygen line core
             oxygen2map     - The array of indices of the second oxygen line core
    OUTPUTS: marginalmap     - An array of the MAP solution for f for each pixel
    EXAMPLE: 
    """    
    dispersion=Dispersion(oxygen1map=oxygen1map, oxygen2map=oxygen2map)#angstroms/wavelength pixel
    lambdanaughts=LambdaNaught(ironmap=ironmap, oxygen1map=oxygen1map, dispersion=dispersion)#wavelenth of iron line core
    alphas=Alpha(lambdanaught=lambdanaughts)#alpha values
    MAPmap=[]
    for i in np.arange(1470,1510,1): #just doing a region currently
        print(i, str(datetime.now()))
        row=[]
        ssI=v9s.SingleStokes(datapath=datapath, stokes='I', row=i) #row stokes I data
        ssV=v9s.SingleStokes(datapath=datapath, stokes='V', row=i) #row stokes V data
        for j in np.arange(370,410,1): #just doing a region currently
            if lambdanaughts[i][j]==0:
                row.append(0)
            else:
                MAP=fpixelMAP(Islicedata=ssI, Vslicedata=ssV, col=j, dispersion=dispersion[i][j], alpha=alphas[i][j], ironloc=ironmap[i][j], sigmamax=3000., sigmamin=1., integration='quad')
                row.append(MAP)
        MAPmap.append(row)
    return MAPmap                
    
def Bpixelmarginal(Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000, sigmamin=1, integration='quad'):
    """NAME: Bpixelmarginal 
    PURPOSE: Returns the pdf of B for a pixel
    INPUTS:  lambdanaughts  - The array of wavelengths of the calcium absorption line core
             Islicedata     - The array of information for stokes I at a horizontal slice
             Vslicedata     - The array of information for stokes V at a horizontal slice (is divided by I)
             row            - The row of the full disk image, from 0 to 2047
             col            - The col of the full disk image, from 0 to 2047
             integration    - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: fprobs         - The probability values for the 50 values of fvalues=np.linspace(0, 1, 50)
    EXAMPLE: See function fMAPmap below for usage.
    """    
    Bvalues=np.linspace(-2000, 2000, 500) #the values of the LOS B field to test the probability of
    Bprobs=[]
    for i in range(len(Bvalues)):
        if integration=='quad':
            Bprobs.append(sp.integrate.quad(func=Bintegrand, a=0., b=1., args=(Bvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))[0])
        if integration=='romberg':
            Bprobs.append(sp.integrate.romberg(function=Bintegrand, a=0., b=1., args=(Bvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc), divmax=20))
    return Bprobs

def BpixelMAP(Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000, sigmamin=1, integration='quad'):
    """NAME: BpixelMAP 
    PURPOSE: Returns the pdf of B for a pixel
    INPUTS:  lambdanaughts  - The array of wavelengths of the calcium absorption line core
             Islicedata     - The array of information for stokes I at a horizontal slice
             Vslicedata     - The array of information for stokes V at a horizontal slice (is divided by I)
             row            - The row of the full disk image, from 0 to 2047
             col            - The col of the full disk image, from 0 to 2047
             integration    - Numerical integration method: 'quad' or 'romberg'. Romberg is more accurate but takes ~1.2 times as long for f.
    OUTPUTS: fprobs         - The probability values for the 50 values of fvalues=np.linspace(0, 1, 50)
    EXAMPLE: See function fMAPmap below for usage.
    """    
    Bvalues=np.linspace(-2000, 2000, 500) #the values of the LOS B field to test the probability of
    Bprobs=[]
    for i in range(len(Bvalues)):
        if integration=='quad':
            Bprobs.append(sp.integrate.quad(func=Bintegrand, a=0., b=1., args=(Bvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))[0])
        if integration=='romberg':
            Bprobs.append(sp.integrate.romberg(function=Bintegrand, a=0., b=1., args=(Bvalues[i],Islicedata, Vslicedata, col, dispersion, alpha, ironloc))[0])
    argmax=np.argmax(np.abs(Bprobs))
    MAP=Bvalues[argmax]
    return MAP

def Bfullmarginal(datapath, ironmap, oxygen1map, oxygen2map):
    """NAME: Bfullmarginal 
    PURPOSE: Returns the pdf for B for each pixel in the disk of the sun
    INPUTS:  datapath        - The final directory in which the FITS files are located
             ironmap         - The array of indices of the iron line core.
             oxygen1map       - The array of indices of the first oxygen line core
             oxygen2map       - The array of indices of the second oxygen line core
    OUTPUTS: marginalmap     - An array for pdfs for B for each pixel
    EXAMPLE: 
    """   
    dispersion=Dispersion(oxygen1map=oxygen1map, oxygen2map=oxygen2map) #angstroms/wavelength pixel
    lambdanaughts=LambdaNaught(ironmap=ironmap, oxygen1map=oxygen1map,dispersion=dispersion) #wavelength of iron line core
    alphas=Alpha(lambdanaught=lambdanaughts) #alpha values for each pixel
    marginalmap=[]
    for i in range(2048):
        row=[]
        ssI=v9s.SingleStokes(datapath=datapath, stokes='I', row=i) #row stokes I data
        ssV=v9s.SingleStokes(datapath=datapath, stokes='V', row=i) #row stokes V data
        for j in range(2048):
            if lambdanaughts[i][j]==0:
                pdf=np.zeros(400)
                pdf[199]=1 #if the pixel is not on the Sun, you want the most likely value to be B=0
                row.append(pdf)
            else:
                row.append(Bpixelmarginal(Islicedata=ssI, Vslicedata=ssV, col=j, dispersion=dispersion[i][j], alpha=alphas[i][j], ironloc=ironmap[i][j], sigmamax=3000., sigmamin=1., integration='quad'))
        marginalmap.append(row)
    return marginalmap

def BMAPmap(datapath, ironmap, oxygen1map, oxygen2map):
    """NAME: BMAPmap 
    PURPOSE: Returns the MAP solution for B for each pixel in the sun. 
    INPUTS:  datapath        - The final directory in which the FITS files are located
             ironmap         - The array of indices of the iron line core.
             oxygen1map       - The array of indices of the first oxygen line core
             oxygen2map       - The array of indices of the second oxygen line core
    OUTPUTS: marginalmap     - An array of the MAP solution for B for each pixel
    EXAMPLE: 
    """    
    dispersion=Dispersion(oxygen1map=oxygen1map, oxygen2map=oxygen2map) #angstroms/wavelength pixel
    lambdanaughts=LambdaNaught(ironmap=ironmap, oxygen1map=oxygen1map,dispersion=dispersion) #wavelength of iron line core
    alphas=Alpha(lambdanaught=lambdanaughts) #alpha values for each pixel
    MAPmap=[]
    for i in np.arange(1470,1490,1): #currently only calculating for a region
        print(i, str(datetime.now()))
        row=[]
        ssI=v9s.SingleStokes(datapath=datapath, stokes='I', row=i) #row stokes I data
        ssV=v9s.SingleStokes(datapath=datapath, stokes='V', row=i) #row stokes V data
        for j in np.arange(1445, 1465,1): #currently only calculating for a region
            if lambdanaughts[i][j]==0:
                row.append(0) #if the pixel is not on the Sun, the most likely value is zero
            else:
                MAP=BpixelMAP(Islicedata=ssI, Vslicedata=ssV, col=j, dispersion=dispersion[i][j], alpha=alphas[i][j], ironloc=ironmap[i][j], sigmamax=3000., sigmamin=1., integration='quad')
                row.append(MAP)
        MAPmap.append(row)
    return MAPmap       

def FastMAP(Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    fmin=0.
    fmax=1.
    nf=11
    bmin=-2000.
    bmax=2000.
    nb=41
    fvals0=np.linspace(0, 1, nf)
    bvals0=np.linspace(-2000, 2000, nb)
    bstep1=bvals0[1]-bvals0[0]
    bs=[bvals0]
    fs=[fvals0]
    bmaps=[]
    fmaps=[]
    bsteps=[]
    fsteps=[]
    posts=[]
    i=0
    while bstep1>1:
       if i==0:
           fvals1, fstep1, fMAP1, bvals1, bstep1, BMAP1, posteriors1=FastMAPiter(fmin, fmax, nf, nf, bmin, bmax, 41, nb, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.)
       else:
           fvals1, fstep1, fMAP1, bvals1, bstep1, BMAP1, posteriors1=FastMAPiter(fs[i][0], fs[i][-1], nf, nf, bs[i][0], bs[i][-1], nb, nb, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.)
       bs.append(bvals1)
       fs.append(fvals1)
       fmaps.append(fMAP1)
       bmaps.append(BMAP1)
       bsteps.append(bstep1)
       fsteps.append(fstep1)
       posts.append(posteriors1)
       i=i+1
       #print(i, bstep1, fstep1)
    return fmaps[-1], bmaps[-1]

def FastMAPiter(fmin, fmax, nf1, nf2, bmin, bmax, nb1, nb2, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.): 
    fvals=np.linspace(fmin, fmax, nf1)
    bvals=np.linspace(bmin, bmax, nb1)+.001
    #bvals=bvals[bvals==0]+.001
    posteriors=np.zeros((nf1, nb1))
    for i in range(nf1):
        for j in range(nb1):
            posteriors[i][j]=JointPosterior(f=fvals[i], B=bvals[j], Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc, sigmamax=3000., sigmamin=1.)
    if np.max(posteriors)>0:
        posteriors=posteriors/np.max(posteriors)
    fmarginals=sp.integrate.simps(posteriors, bvals)
    fMAPloc=np.argmax(fmarginals)
    fMAP=fvals[fMAPloc]
    Bmarginals=sp.integrate.simps(posteriors.T, fvals)
    BMAPloc=np.argmax(Bmarginals)
    BMAP=bvals[BMAPloc] 
    fstep=(fvals[1]-fvals[0])
    fstepnew=4*fstep
    bstep=(bvals[1]-bvals[0])
    bstepnew=(nb2-1)/4*bstep
    if fMAP>fstepnew and fMAP<1-fstepnew:
        fvals=np.linspace(fMAP-fstepnew, fMAP+fstepnew, nf2)
    elif fMAP<=fstepnew:
        fvals=np.linspace(0, fMAP+fstepnew, nf2)
    else:
        fvals=np.linspace(fMAP-fstepnew, 1, nf2)
    bvals=np.linspace(BMAP-bstepnew, BMAP+bstepnew, nb2)
    return fvals, fstep, fMAP, bvals, bstep, BMAP, posteriors