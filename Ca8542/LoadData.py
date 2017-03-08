# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:22:10 2016

@author: jhamer
"""

import os
os.chdir('/Users/jhamer/Research/Scripts/Ca8542/')
import vhsDataHandler as vhs
#import FourParamWFA as FPW
import WeakFieldApprox as WFA
from astropy.io import fits
os.chdir('/Users/jhamer/Research/Data/')
datapath='k4vhs160409t174101_oid114602236001741_cleaned'
ls=['I', 'Q', 'U', 'V']
stokes=[]
#row=1170
#col=430
#for l in ls:
#    stokes.append(v9s.SingleStokes(datapath, l, row))
#ca=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/CalciumAbsorptionCoreIndex.fits')
#wa=fits.getdata('/Users/jhamer/Research/Output/k4vhs160409t174101/WaterAbsorptionCoreIndex.fits')
#lambdanaught=WFA.LambdaNaught(fe, ox1, dispersion)
#alpha=WFA.Alpha(lambdanaught)
#fmapmap, bmapmap=WFA.fastMAPmap(datapath, fe, ox1, ox2)
#beta=FPW.Beta(lambdanaught)
#return datapath, stokes, fe, ox1, ox2, dispersion, lambdanaught, alpha, beta
#nf=10
#nBlos=20
#nBperp=10
#nchi=10
#f, Blos, Bperp, X=FPW.pixelmarginals(10, 20, 10, 10, stokes[0], stokes[1], stokes[2], stokes[3], col, dispersion[row][col], alpha[row][col], beta[row][col], fe[row][col])
'''fvalues=np.linspace(0,1,nf)
Blosvalues=np.linspace(-2000, 2000, nBlos)+.01
Bperpvalues=np.linspace(0, 2000, nBperp)+.01
Xvalues=np.linspace(0, 2.*np.pi, nchi)
post=FPW.PosteriorMatrix(nf=nf, nBlos=nBlos, nBperp=nBperp, nchi=nchi, Islicedata=stokes[0], Qslicedata=stokes[1], Uslicedata=stokes[2], Vslicedata=stokes[3], col=col, dispersion=dispersion[row][col], alpha=alpha[row][col], beta=beta[row][col], ironloc=fe[row][col])
postT=post.T
int1=sp.integrate.simps(post, Xvalues)
transint1=sp.integrate.simps(postT, fvalues)
int2=sp.integrate.simps(int1, Bperpvalues)
'''