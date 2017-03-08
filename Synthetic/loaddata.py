# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:17:27 2016

@author: jhamer
"""

import os
os.chdir('/Users/jhamer/Research/Scripts/Synthetic/')
import v9sDataHandler as v9s
import WeakFieldApprox as WFA
#import FourParamWFA as FPW
import CoreLocation as CL
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.io import fits
os.chdir('/Users/jhamer/Research/Data/')
datapath='k4v9s160617t000000_1024_wfa_noscat'
ls=['I', 'Q', 'U', 'V']
stokes=[]
for l in ls:
    stokes.append(v9s.SingleStokes(datapath, l))
fe, ox1, ox2=CL.IndexMaps(datapath, stokes[0][70])
dispersion=WFA.Dispersion(ox1, ox2)
lambdanaught=WFA.LambdaNaught(fe, ox1, dispersion)
alpha=WFA.Alpha(lambdanaught)
#beta=FPW.Beta(lambdanaught)
Islicedata=stokes[0]
#Qslicedata=stokes[1]
#Uslicedata=stokes[2]
Vslicedata=stokes[3]
col=256
dispersion=dispersion[col]
alpha=alpha[col]
#beta=beta[col]
ironloc=fe[col]
params=fits.getdata('/Users/jhamer/Downloads/synth_spec_params.fits')
#cols=np.arange(0, 2048, 3)
#fmaps=[]
#blosmaps=[]
#btransmaps=[]
#chimaps=[]
'''
t1=str(datetime.now())
for i in range(len(cols)):
    col=cols[i]
    print(col)
    if np.max(stokes[0][70][col])>0:  
        #fMAP, BlosMAP, BtransMAP, chiMAP=FPW.FastMAP(stokes[0], stokes[1], stokes[2], stokes[3], col, dispersion[col], alpha[col], beta[col], fe[col])
        fMAP, BlosMAP=WFA.FastMAP(stokes[0], stokes[3], col, dispersion[col], alpha[col], fe[col])
        fmaps.append(fMAP)
        blosmaps.append(BlosMAP)
        #btransmaps.append(BtransMAP)
        #chimaps.append(chiMAP)
    else:
        fmaps.append(0)
        blosmaps.append(0)
        #btransmaps.append(0)
        #chimaps.append(0)
t2=str(datetime.now())
print('start = '+t1,'finish = '+t2)

f=plt.figure()
plt.plot(cols, blosmaps, label='Calculated')
plt.plot(cols, params[1][0:-1:3], label='Actual')
plt.plot.set_ylabel('LOS B [G]')
plt.savefig('/Users/jhamer/Research/2DBlos_FastMAP_Solutions.pdf')
'''
'''
colshdu=fits.PrimaryHDU(cols)
fmapshdu=fits.PrimaryHDU(fmaps)
blosmapshdu=fits.PrimaryHDU(blosmaps)
btransmapshdu=fits.PrimaryHDU(btransmaps)
chimapshdu=fits.PrimaryHDU(chimaps)
colshdu.writeto('/Users/jhamer/Research/cols.fits')
fmapshdu.writeto('/Users/jhamer/Research/fmaps.fits')
blosmapshdu.writeto('/Users/jhamer/Research/blosmaps.fits')
btransmapshdu.writeto('/Users/jhamer/Research/btransmaps.fits')
chimapshdu.writeto('/Users/jhamer/Research/chimaps.fits')

f, ax=plt.subplots(4,1)
ax[0].plot(cols, fmaps, label='Calculated')
ax[0].plot(cols, params[0][0:-1:4], label='Actual')
ax[0].set_ylabel('f')
ax[1].plot(cols, blosmaps, label='Calculated')
ax[1].plot(cols, params[1][0:-1:4], label='Actual')
ax[1].set_ylabel('LOS B [G]')
ax[2].plot(cols, btransmaps, label='Calculated')
ax[2].plot(cols, params[2][0:-1:4], label='Actual')
ax[2].set_ylabel('Trans. B [G]')
ax[3].plot(cols, chimaps, label='Calculated')
ax[3].plot(cols, params[3][0:-1:4], label='Actual')
ax[3].set_ylabel('chi [rad]')
plt.savefig('/Users/jhamer/Research/4D_FastMAP_Solutions.png')
'''
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
       fvals1, fstep1, fMAP1, bvals1, bstep1, BMAP1, posteriors1=WFA.FastMAPiter(fmin, fmax, nf, nf, bmin, bmax, 41, nb, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.)
   else:
       fvals1, fstep1, fMAP1, bvals1, bstep1, BMAP1, posteriors1=WFA.FastMAPiter(fs[i][0], fs[i][-1], nf, nf, bs[i][0], bs[i][-1], nb, nb, Islicedata, Vslicedata, col, dispersion, alpha, ironloc, sigmamax=3000., sigmamin=1.)
   bs.append(bvals1)
   fs.append(fvals1)
   fmaps.append(fMAP1)
   bmaps.append(BMAP1)
   bsteps.append(bstep1)
   fsteps.append(fstep1)
   posts.append(posteriors1)
   i=i+1
   print(i, bstep1, fstep1)

'''
fmin=0.
fmax=1.
nf=11
blosmin=-2000.
blosmax=2000.
nblos=21
btransmin=0.
btransmax=2000.
nbtrans=21
chimin=0.
chimax=np.pi
nchi=11
fvals0=np.linspace(fmin, fmax, nf)
blosvals0=np.linspace(blosmin, blosmax, nblos)+.0001
btransvals0=np.linspace(btransmin, btransmax, nbtrans)
chivals0=np.linspace(chimin,chimax, nchi)
blosstep=blosvals0[1]-blosvals0[0]
bloss=[blosvals0]
fs=[fvals0]
btranss=[btransvals0]
chis=[chivals0]
blosmaps=[]
fmaps=[]
btransmaps=[]
chimaps=[]
blossteps=[]
fsteps=[]
btranssteps=[]
chisteps=[]
posts=[]
i=0
print(str(datetime.now()))
while blosstep>4:
   if i==0:
       fvals, fstep, fMAP, blosvals, blosstep, BlosMAP, btransvals, btransstep, BtransMAP, chivals, chistep, chiMAP, posteriors=FPW.FastMAPiter(fmin, fmax, nf, nf, blosmin, blosmax, nblos, nblos, btransmin, btransmax, nbtrans, nbtrans, chimin, chimax, nchi, nchi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.)
   else:
       fvals, fstep, fMAP, blosvals, blosstep, BlosMAP, btransvals, btransstep, BtransMAP, chivals, chistep, chiMAP, posteriors=FPW.FastMAPiter(fs[i][0], fs[i][-1], nf, nf, bloss[i][0], bloss[i][-1], nblos, nblos, btranss[i][0], btranss[i][-1], nbtrans, nbtrans, chis[i][0], chis[i][-1], nchi, nchi, Islicedata, Qslicedata, Uslicedata, Vslicedata, col, dispersion, alpha, beta, ironloc, sigmamax=3000., sigmamin=1.)
   bloss.append(blosvals)
   fs.append(fvals)
   btranss.append(btransvals)
   chis.append(chivals)
   fmaps.append(fMAP)
   blosmaps.append(BlosMAP)
   btransmaps.append(BtransMAP)
   chimaps.append(chiMAP)
   blossteps.append(blosstep)
   fsteps.append(fstep)
   btranssteps.append(btransstep)
   chisteps.append(chistep)
   posts.append(posteriors)
   i=i+1
   print(i, blosstep)
print(str(datetime.now()))
'''