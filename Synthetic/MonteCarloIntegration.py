# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:12:27 2016

@author: jhamer
"""

import itertools as it
import numpy as np

def sampler(f=False, Blos=False, Bperp=False, chi=False):
    while True:
        params={}
        if f==True:
            params['f']=np.random.uniform(0.,1.)
        elif type(f)==float:
            params['f']=f
        if Blos==True:
            params['Blos']=np.random.uniform(-2000.,2000.)
        elif type(Blos)==float:
            params['Blos']=Blos
        if Bperp==True:
            params['Bperp']=np.random.uniform(0.,2000.)
        elif type(Bperp)==float:
            params['Bperp']=Bperp
        if chi==True:
            params['chi']=np.random.uniform(0., 2.*np.pi)
        elif type(chi)==float:
            params['chi']=chi
        yield params
'''

def B_MCI_integrand(param_dict, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    f=param_dict['f']
    B=param_dict['Blos']
    A1, A2, A3, N=A1A2A3(Islicedata=Islicedata, Vslicedata=Vslicedata, f=f, B=B, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc)
    sigma_d=VminusdIvar(Islicedata=Islicedata, Vslicedata=Vslicedata, dispersion=dispersion, alpha=alpha, f=f, B=B, col=col)
    exp=np.exp(-(A1-(2.*alpha*f*B*A2)+((alpha*f*B)**2.)*A3))
    return .5*(2.*np.pi)**(-N/2.)*sigma_d**(-N)*(1./B)*StandardRectangleFunction(f=f)*exp*ErfDiff(B=B, sigmamax=sigmamax, sigmamin=sigmamin)

def B_MCI_marg(nB, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.):
    Bvalues=np.linspace(-2000., 2000., nB)
    Bvalues=Bvalues[Bvalues!=0]
    Bprobs=[]
    for i in range(nB):
        prob=MCI.MCIntegrate(integrand=B_MCI_integrand, sampler=MCI.sampler, Blos=Bvalues[i], f=True, Islicedata=Islicedata, Vslicedata=Vslicedata, col=col, dispersion=dispersion, alpha=alpha, ironloc=ironloc, sigmamax=sigmamax, sigmamin=sigmamin)
        Bprobs.append(prob)
    return (Bvalues, Bprobs)
'''

#def MCIntegrate(integrand, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1., f=False, Blos=False, Bperp=False, chi=False, scale=1.0, n=1000):
def MCIntegrate(integrand, f=False, Blos=False, Bperp=False, chi=False, scale=1.0, n=1000, *args):
    fsum=0.0
    fsum_of_squares=0.0
    for i in it.islice(sampler(f, Blos, Bperp, chi), n):
        #f=integrand(i, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax, sigmamin)
        f=integrand(i, *args)
        fsum+=f
        fsum_of_squares+=(f**2.)
    expectation=fsum/n
    expectation_of_sq=fsum_of_squares/n
    variance=(expectation_of_sq-expectation**2.)
    return ((scale*expectation, scale*np.sqrt(variance/n)))
