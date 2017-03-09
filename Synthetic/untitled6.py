# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:49:43 2016

@author: jhamer
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def MultiImagePlot(images, title, subtitles, colorbarlabels):
    if len(images)<=3:
        f, axes=plt.subplots(1,len(images))
        ims=[]
        divs=[]
        caxs=[]
        for i in range(len(images)):
            ims.append(axes[i].imshow(images[i], origin='lower'))
            divs.append(make_axes_locatable(axes[i]))
            caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
            axes[i].set_title(subtitles[i])
            plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
        f.suptitle(title, fontsize=18)
    elif len(images)==4:
        f, axes=plt.subplots(2, 2)
        ims=[]
        divs=[]
        caxs=[]
        for i in range(len(images)):
            if i%2==0:
                ims.append(axes[i/2][0].imshow(images[i], origin='lower'))
                divs.append(make_axes_locatable(axes[i/2][0]))
                caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
                axes[i].set_title(subtitles[i])
                plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
            if i%2==1:                
                ims.append(axes[i/2][1].imshow(images[i], origin='lower'))
                divs.append(make_axes_locatable(axes[i/2][1]))
                caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
                axes[i].set_title(subtitles[i])
                plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
        f.suptitle(title, fontsize=18)
    else:
        f, axes=plt.subplots(int(len(images)/3), 3)
        ims=[]
        divs=[]
        caxs=[]
        for i in range(len(images)):
            if i%3==0:
                ims.append(axes[i/3][0].imshow(images[i], origin='lower'))
                divs.append(make_axes_locatable(axes[i/3][0]))
                caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
                axes[i].set_title(subtitles[i])
                plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
            if i%3==1:                
                ims.append(axes[i/3][1].imshow(images[i], origin='lower'))
                divs.append(make_axes_locatable(axes[i/3][1]))
                caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
                axes[i].set_title(subtitles[i])
                plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
            if i%3==2:                
                ims.append(axes[i/3][2].imshow(images[i], origin='lower'))
                divs.append(make_axes_locatable(axes[i/3][2]))
                caxs.append(divs[i].append_axes('right', size='10%', pad=0.05))
                axes[i].set_title(subtitles[i])
                plt.colorbar(ims[i], caxs[i], label=colorbarlabels[i])
        f.suptitle(title, fontsize=18)
    return f, axes