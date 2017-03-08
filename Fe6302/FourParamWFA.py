# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:24:57 2016

@author: jhamer
"""

import v9sDataHandler as v9s
import numpy as np
import scipy as sp
import scipy.special as sps
from datetime import datetime
import mcint

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
    lambdanaughts=6302.0005*np.ones((len(ironmap), len(ironmap))) #base of 6302.0005A for every pixel #for real data change to np.ones((len(ironmap), len(ironmap)))
    distance=ironmap-oxygen1map #number of indices between Fe6302 Line core and the first terrestrial oxygen line
    wavelengthdistance=distance*dispersion #converts index distance to distance in Angstroms
    lambdanaughts=lambdanaughts+wavelengthdistance #adds the distance to the base 6302.0005 location of the first oxygen line
    return lambdanaughts #array of locations of iron line core in angstroms

def Sigma_n(Islicedata, Qslicedata, Uslicedata, Vslicedata, dispersion, alpha, beta, f, Blos, Bperp, col):
    """NAME: Sigma_n
    PURPOSE: Returns noise of data
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta        - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047
    OUTPUTS: sigma        - The noise to be plugged into the Generalized Posterior
    EXAMPLE: 
    """
    Iflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Islicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Qflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Qslicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Uflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Uslicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Vflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Vslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel
    Icontinuum=Iflux[97:115] #picks out the wavelengths at which there is continuum
    Qcontinuum=Qflux[97:115] #picks out the wavelengths at which there is continuum
    Ucontinuum=Uflux[97:115] #picks out the wavelengths at which there is continuum
    Vcontinuum=Vflux[97:115] #picks out the wavelengths at which there is continuum
    deriv=np.gradient(Icontinuum)/dispersion #the wavelength derivative of Intensity, calculated by chain rule: dI/dpixel * dpixel/dAngstrom
    derivsq=np.gradient(deriv)/dispersion #the second wavelength derivative of Intensity, calculated by chain rule: dI/dpixel * dpixel/dAngstrom
    dIvar=np.var(deriv) #variance of the derivative
    dIsqvar=np.var(derivsq) #variance of the second derivative
    Vvar=np.var(Vcontinuum) #variance in V
    Qvar=np.var(Qcontinuum) #variance in Q
    Uvar=np.var(Ucontinuum) #variance in U
    #cov=np.cov((Vcontinuum,deriv))[0][1] #the covariance is a matrix, the terms on the opposite diagonal are the cross terms
    #var=Vvar+(alpha*f*B)**2.*dIvar-2.*(alpha*f*B)*cov #based on Var(X+Y)=Var(X)+Var(Y)-2*Cov(X,Y) - for Var(dI) the constants come out twice
    var=Vvar+Qvar+Uvar+(alpha*f*Blos)**2.*dIvar+(beta*f*Bperp**2.)**2.*dIsqvar
    #var=Vvar+Qvar+Uvar #simplest case
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
    
def Beta(lambdanaught):
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
    Geff=1 #GET ACTUAL VALUE
    alpha=-5.45*10**-26*Geff*(lambdanaught**4.) #defined on bottom of page 2 of Asensios Ramos 2011
    return alpha

def Avalues(Islicedata, Qslicedata, Uslicedata, Vslicedata, f, Blos, Bperp, col, dispersion, alpha, beta, ironloc):
    """NAME: A1A2A3 
    PURPOSE: Returns A1, A2, A3, N defined on page 3 of Asensios
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta        - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
    OUTPUTS: A1           - 
             A2           - 
             A3           - 
             A4           - 
             A5           - 
             A6           - 
             N            - N is the number of wavelength pixels around the core region included
    EXAMPLE: 
    """
    Iflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Islicedata, xpos=col)) #loads in array of stokes I intensities over wavelength for a spatial pixel
    Qflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Qslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel    
    Uflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Uslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel    
    Vflux=np.array(v9s.LoadedDataSpectrum(Islicedata=Islicedata, slicedata=Vslicedata, xpos=col)) #loads in array of stokes V intensities over wavelength for a spatial pixel    
    ironloc=np.int(ironloc) #the iron absorption core loc for the single pixel
    core=np.arange(ironloc-2, ironloc+3,1) #the indices of the region of the absorption line in the spectrum
    Icore=Iflux[core] #selects out the region of iron absorption in the spectrum
    Qcore=Qflux[core] #selects out the region of iron absorption in the spectrum
    Ucore=Uflux[core] #selects out the region of iron absorption in the spectrum
    Vcore=Vflux[core] #selects out the region of iron absorption in the spectrum
    gradI=np.gradient(Icore)/dispersion #dI/dlambda
    gradsqI=np.gradient(gradI)/dispersion #dI/dlambda
    sigma=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    A1=np.ndarray.sum(Vcore**2.+Qcore**2.+Ucore**2.)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos
    A2=np.ndarray.sum(gradI**2.)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    A3=np.ndarray.sum(gradsqI**2.)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    A4=np.ndarray.sum(Vcore*gradI)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    A5=np.ndarray.sum(Qcore*gradsqI)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    A6=np.ndarray.sum(Ucore*gradsqI)/(2.*sigma**2.) #as defined on page 3 of Asensios Ramos, but I keep alpha outside it
    N=len(core) #number of wavelength pixels included in summation
    return A1, A2, A3, A4, A5, A6, N
    
def StandardRectangleFunction(arg):
    """NAME: StandardRectangleFunction 
    PURPOSE: Prior distribution for fill factor, f.
    INPUTS:  f   - The value of the filling factor, from 0 to 1.
    OUTPUTS: 0, 1, or .5 depending on value of f
    EXAMPLE: 
            In [6]: srf=StandardRectangleFunction(0.5)
            In [7]: srf
            Out[7]: 1
    """
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

def ChiIntegrand(f, Blos, Bperp, chi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: ChiIntegrand 
    PURPOSE: Returns the integrand for triple numerical integration to find the posterior distribution of Chi
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta         - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
             sigmamin     - The min possible noise value on B LOS
             sigmamax     - The max possible noise value on B LOS
    OUTPUTS: post         - The generalized posterior for f, B_LOS, Bperp, X given known noise variance. To be integrated numerically in Xpixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    NOTES:   scipy.integrate.nquad integrates over the first three arguments with the first as the innermost integral, so the first argument passed in the integration function is the value of chi for which probability is being calculated
    """
    A1, A2, A3, A4, A5, A6, N=Avalues(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, f=f, Blos=Blos, Bperp=Bperp, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc)
    sigma_n=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    exp1=np.e**(-(A1+A2*(alpha*f*Blos)**2.+A3*(beta*f*Bperp**2.)**2.-2.*A4*alpha*f*Blos-2.*(A5*np.cos(2.*chi)+A6*np.sin(2.*chi))*beta*f*(Bperp**2.)))
    expdiff=np.e**(-(Bperp**2./(2.*sigmamax**2.)))-np.e**(-(Bperp**2./(2.*sigmamin**2.)))
    erf=ErfDiff(B=Blos, sigmamax=sigmamax, sigmamin=sigmamin)
    srf1=StandardRectangleFunction(f-0.5)
    srf2=StandardRectangleFunction((chi-np.pi)/(2.*np.pi))
    post=0.5*srf1*srf2*((2.*np.pi)**((-3.*N+1.)/2))*(sigma_n**(-3.*N))*exp1*(1./(Blos*Bperp))*erf*expdiff
    return post
    
def BperpIntegrand(f, Blos, chi, Bperp, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: BperpIntegrand 
    PURPOSE: Returns the integrand for triple numerical integration to find the posterior distribution of Bperp
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta         - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
             sigmamin     - The min possible noise value on B LOS
             sigmamax     - The max possible noise value on B LOS
    OUTPUTS: post         - The generalized posterior for f, B_LOS, Bperp, X given known noise variance. To be integrated numerically in Bperppixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    NOTES:   scipy.integrate.nquad integrates over the first three arguments with the first as the innermost integral, so the first argument passed in the integration function is the value of Bperp for which probability is being calculated
    """
    A1, A2, A3, A4, A5, A6, N=Avalues(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, f=f, Blos=Blos, Bperp=Bperp, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc)
    sigma_n=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    exp1=np.e**(-(A1+A2*(alpha*f*Blos)**2.+A3*(beta*f*Bperp**2.)**2.-2.*A4*alpha*f*Blos-2.*(A5*np.cos(2.*chi)+A6*np.sin(2.*chi))*beta*f*(Bperp**2.)))
    expdiff=np.e**(-(Bperp**2./(2.*sigmamax**2.)))-np.e**(-(Bperp**2./(2.*sigmamin**2.)))
    erf=ErfDiff(B=Blos, sigmamax=sigmamax, sigmamin=sigmamin)
    srf1=StandardRectangleFunction(f-0.5)
    srf2=StandardRectangleFunction((chi-np.pi)/(2.*np.pi))
    post=0.5*srf1*srf2*((2.*np.pi)**((-3.*N+1.)/2))*(sigma_n**(-3.*N))*exp1*(1./(Blos*Bperp))*erf*expdiff
    return post
    
def BlosIntegrand(f, Bperp, chi, Blos, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: BlosIntegrand 
    PURPOSE: Returns the integrand for triple numerical integration to find the posterior distribution of Blos
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta         - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
             sigmamin     - The min possible noise value on B LOS
             sigmamax     - The max possible noise value on B LOS
    OUTPUTS: post         - The generalized posterior for f, B_LOS, Bperp, X given known noise variance. To be integrated numerically in Blospixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    NOTES:   scipy.integrate.nquad integrates over the first three arguments with the first as the innermost integral, so the first argument passed in the integration function is the value of Blos for which probability is being calculated
    """
    A1, A2, A3, A4, A5, A6, N=Avalues(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, f=f, Blos=Blos, Bperp=Bperp, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc)
    sigma_n=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    exp1=np.e**(-(A1+A2*(alpha*f*Blos)**2.+A3*(beta*f*Bperp**2.)**2.-2.*A4*alpha*f*Blos-2.*(A5*np.cos(2.*chi)+A6*np.sin(2.*chi))*beta*f*(Bperp**2.)))
    expdiff=np.e**(-(Bperp**2./(2.*sigmamax**2.)))-np.e**(-(Bperp**2./(2.*sigmamin**2.)))
    erf=ErfDiff(B=Blos, sigmamax=sigmamax, sigmamin=sigmamin)
    srf1=StandardRectangleFunction(f-0.5)
    srf2=StandardRectangleFunction((chi-np.pi)/(2.*np.pi))
    post=0.5*srf1*srf2*((2.*np.pi)**((-3.*N+1.)/2))*(sigma_n**(-3.*N))*exp1*(1./(Blos*Bperp))*erf*expdiff
    return post
    
def fIntegrand(Blos, Bperp, chi, f, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: fIntegrand 
    PURPOSE: Returns the integrand for triple numerical integration to find the posterior distribution of f
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta         - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
             sigmamin     - The min possible noise value on B LOS
             sigmamax     - The max possible noise value on B LOS
    OUTPUTS: post         - The generalized posterior for f, B_LOS, Bperp, X given known noise variance. To be integrated numerically in fpixelmarginal
    EXAMPLE: See function Bpixelmarginal below for usage.
    NOTES:   scipy.integrate.nquad integrates over the first three arguments with the first as the innermost integral, so the first argument passed in the integration function is the value of f for which probability is being calculated
    """
    A1, A2, A3, A4, A5, A6, N=Avalues(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, f=f, Blos=Blos, Bperp=Bperp, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc)
    sigma_n=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    exp1=np.e**(-(A1+A2*(alpha*f*Blos)**2.+A3*(beta*f*Bperp**2.)**2.-2.*A4*alpha*f*Blos-2.*(A5*np.cos(2.*chi)+A6*np.sin(2.*chi))*beta*f*(Bperp**2.)))
    expdiff=np.e**(-(Bperp**2./(2.*sigmamax**2.)))-np.e**(-(Bperp**2./(2.*sigmamin**2.)))
    erf=ErfDiff(B=Blos, sigmamax=sigmamax, sigmamin=sigmamin)
    srf1=StandardRectangleFunction(f-0.5)
    srf2=StandardRectangleFunction((chi-np.pi)/(2.*np.pi))
    post=0.5*srf1*srf2*((2.*np.pi)**((-3.*N+1.)/2))*(sigma_n**(-3.*N))*exp1*(1./(Blos*Bperp))*erf*expdiff
    return post
    
def JointPosterior(f, Blos, Bperp, chi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    """NAME: JointPosterior 
    PURPOSE: Returns the integrand for triple numerical integration to find the posterior distribution of Blos
    INPUTS:  Islicedata   - The array of information for stokes I at a horizontal slice
             Qslicedata   - The array of information for stokes Q at a horizontal slice 
             Uslicedata   - The array of information for stokes U at a horizontal slice 
             Vslicedata   - The array of information for stokes V at a horizontal slice 
             dispersion   - The number of angstroms/wavelength pixel
             alpha        - Constant alpha returned by function Alpha above  
             beta         - Constant alpha returned by function Beta above  
             f            - The value of the filling factor
             Blos         - The value of the LOS B field    
             Bperp        - The value of the perpendicular B field    
             col          - The col of the full disk image, from 0 to 2047 
             ironloc      - Index of iron absorption core         
             sigmamin     - The min possible noise value on B LOS
             sigmamax     - The max possible noise value on B LOS
    OUTPUTS: post         - The generalized posterior for f, B_LOS, Bperp, X given known noise variance. To be integrated numerically in Blospixelmarginal
    EXAMPLE: See function PosteriorMatrix below for usage.
    NOTES:   I use this function when manually calculating the posterior value for a specific f, Blos, Bperp, X
    """
    A1, A2, A3, A4, A5, A6, N=Avalues(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, f=f, Blos=Blos, Bperp=Bperp, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc)
    sigma_n=Sigma_n(Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, beta=beta, f=f, Blos=Blos, Bperp=Bperp, col=col)
    exp1=np.e**(-(A1+A2*(alpha*f*Blos)**2.+A3*(beta*f*Bperp**2.)**2.-2.*A4*alpha*f*Blos-2.*(A5*np.cos(2.*chi)+A6*np.sin(2.*chi))*beta*f*(Bperp**2.)))
    expdiff=np.e**(-(Bperp**2./(2.*sigmamax**2.)))-np.e**(-(Bperp**2./(2.*sigmamin**2.)))
    erf=ErfDiff(B=Blos, sigmamax=sigmamax, sigmamin=sigmamin)
    srf1=StandardRectangleFunction(f-0.5)
    srf2=StandardRectangleFunction((chi-np.pi)/(2.*np.pi))
    post=0.5*srf1*srf2*((2.*np.pi)**((-3.*N+1.)/2))*(sigma_n**(-3.*N))*exp1*(1./(Blos*Bperp))*erf*expdiff
    return post

def PosteriorMatrix(nf, nBlos, nBperp, nchi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    fvals=np.linspace(0, 1, nf)
    Blosvals=np.linspace(-2000, 2000, nBlos)+.01
    Bperpvals=np.linspace(0, 2000, nBperp)+.01
    chivals=np.linspace(0, 2.*np.pi, nchi)
    posteriors=np.zeros((nf, nBlos, nBperp, nchi))
    for i in range(nf):
        for j in range(nBlos):
            for k in range(nBperp):
                for l in range(nchi):
                    posteriors[i][j][k][l]=JointPosterior(f=fvals[i], Blos=Blosvals[j], Bperp=Bperpvals[i], chi=chivals[i], Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc, sigmamax=sigmamax, sigmamin=sigmamin)
    if np.max(posteriors)>0:
        posteriors=posteriors/np.max(posteriors)
    return posteriors

def pixelmarginals(nf, nBlos, nBperp, nchi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    fvalues=np.linspace(0,1,nf)
    Blosvalues=np.linspace(-2000, 2000, nBlos)+.01
    Bperpvalues=np.linspace(0, 2000, nBperp)+.01
    Xvalues=np.linspace(0, 2.*np.pi, nchi)
    post=PosteriorMatrix(nf=nf, nBlos=nBlos, nBperp=nBperp, nchi=nchi, Islicedata=Islicedata, Qslicedata=Qslicedata, Uslicedata=Uslicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc, sigmamax=sigmamax, sigmamin=sigmamin)
    postT=post.T
    int1=sp.integrate.simps(post, Xvalues)
    transint1=sp.integrate.simps(postT, fvalues)
    int2=sp.integrate.simps(int1, Bperpvalues)
    transint2=sp.integrate.simps(transint1, Blosvalues)
    fmarg=sp.integrate.simps(int2, Blosvalues)
    if np.max(fmarg)>0:
        fmarg=fmarg/np.max(fmarg)
    Blosmarg=sp.integrate.simps(int2.T, fvalues)
    if np.max(Blosmarg)>0:
        Blosmarg=Blosmarg/np.max(Blosmarg)
    Xmarg=sp.integrate.simps(transint2, Bperpvalues)
    if np.max(Xmarg)>0:
        Xmarg=Xmarg/np.max(Xmarg)
    Bperpmarg=sp.integrate.simps(transint2.T, Xvalues)
    if np.max(Bperpmarg)>0:
        Bperpmarg=Bperpmarg/np.max(Bperpmarg)
    return (fvalues, fmarg), (Blosvalues, Blosmarg), (Bperpvalues, Bperpmarg), (Xvalues, Xmarg)
    
def pixelMAPs(nf, nBlos, nBperp, nchi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.):
    fvalues=np.linspace(0,1,nf)
    Blosvalues=np.linspace(-2000, 2000, nBlos)
    Bperpvalues=np.linspace(0, 2000, nBperp)
    Xvalues=np.linspace(0, 2.*np.pi, nchi)
    post=PosteriorMatrix(nf=nf, nBlos=nBlos, nBperp=nBperp, nchi=nchi ,Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, beta=beta, ironloc=ironloc, sigmamax=3000., sigmamin=1.)
    postT=post.T
    int1=sp.integrate.simps(post, Xvalues)
    transint1=sp.integrate.simps(postT, fvalues)
    int2=sp.integrate.simps(int1, Bperpvalues)
    transint2=sp.integrate.simps(transint1, Blosvalues)
    fmarg=sp.integrate.simps(int2, Blosvalues)
    Blosmarg=sp.integrate.simps(int2.T, fvalues)
    Xmarg=sp.integrate.simps(transint2, Bperpvalues)
    Bperpmarg=sp.integrate.simps(transint2.T, Xvalues)
    fargmax=np.argmax(fmarg)
    Blosargmax=np.argmax(Blosmarg)
    Bperpargmax=np.argmax(Bperpmarg)
    Xargmax=np.argmax(Xmarg)
    fMAP=fvalues[fargmax]
    BlosMAP=Blosvalues[Blosargmax]
    BperpMAP=Bperpvalues[Bperpargmax]
    XMAP=Xvalues[Xargmax]
    return fMAP, BlosMAP, BperpMAP, XMAP
    
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

