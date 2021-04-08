#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from skimage.util import invert


def plot_image(stage, chains=None, axis_off=True, inverted=False, title=''):
    if inverted:
        stage = invert(stage)
    fig,ax = plt.subplots(dpi=300)
    ax.imshow(stage, cmap=plt.cm.gray)
    if axis_off:
        ax.set_axis_off()    
    if title:
        ax.set_title(title)
    if chains:
        for chain in chains: # add chains to image plot
            yy,xx = np.array(chain).T
            plt.plot(xx,yy,lw=0.2)
    plt.show() 


def draw_tiles(polygons, colors, h, w, background_brightness=0.25, return_svg=None,
               chains=None, axis_off=True, title=''):
    fig,ax = plt.subplots(dpi=500)
    if axis_off:
        ax.set_axis_off()
    if title:
        ax.set_title(title)
    if h and w:
        ax.imshow(np.ones((h,w))*background_brightness,cmap=plt.cm.gray, vmin=0, vmax=1 ) # 0.2 is quite dark
    else:
        ax.invert_yaxis()
        ax.autoscale()

    if return_svg:
        svg = """<?xml version="1.0" encoding="UTF-8"?>
                <svg xmlns="http://www.w3.org/2000/svg" version="1.1" baseProfile="full">
                <rect width="100%" height="100%" fill="dimgrey"/>
                """
        
    for j,p in enumerate(polygons): #+

        if colors:
            color = colors[j]
            edgecolor = None
        else:
            color = 'silver'
            edgecolor = 'black'
    
        x,y = p.exterior.xy; #plt.plot(x,y)
        corners = np.array(p.exterior.coords.xy).T
        stein = patches.Polygon(corners, edgecolor=edgecolor, lw=0.3, facecolor=color)# facecolor=color)    
        ax.add_patch(stein)
        
        if return_svg:
            svg_koords = ' '.join([f'{x:.1f} {y:.1f}' for x,y in list(p.exterior.coords)])
            color = (color*255).round()
            color_svg = 'fill="rgb('+','.join([str(int(c)) for c in color])+')"'
            svg_p = f'<polygon points="{svg_koords}" {color_svg}/>\n'
            svg += svg_p
        
    if chains:
        for chain in chains:
            yy,xx = np.array(chain).T
            ax.plot(xx,yy,lw=0.7) #, c='w'
    plt.show()
    
    return svg+'</svg>' if return_svg else None


def statistics(polygons):
    corner_counts = []
    for j,p in enumerate(polygons):
        count = np.array(p.exterior.coords).shape[0]
        corner_counts += [ count ]
    fig,ax = plt.subplots(dpi=300)
    ax.hist(corner_counts, bins=range(3,10), rwidth=0.9, align='left')
    ax.set_title(f'{len(polygons)} polygons; corners: min={min(corner_counts)}, max={max(corner_counts)}, avg={sum(corner_counts)/len(corner_counts):.1f}')
    ax.set_ylabel('number of tiles')
    ax.set_xlabel('number of corners')






