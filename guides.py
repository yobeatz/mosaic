#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from skimage import draw
from scipy.ndimage import label, morphology
import copy
import time
import skimage as sk
import matplotlib as mpl
import plotting
mpl.rcParams['figure.dpi'] = 300



def pixellines_to_ordered_points(matrix, half_tile):
    # break guidelines into chains and order the pixel for all chain

    # import cv2
    # chains3 = []

    matrix = sk.morphology.skeletonize(matrix) # nicer lines, better results
    matrix_labeled, chain_count = label(matrix, structure=[[1,1,1], [1,1,1], [1,1,1]]) # find chains
    chains = []
    for i_chain in range(1,chain_count):
        pixel = copy.deepcopy(matrix_labeled)
        pixel[pixel!=i_chain] = 0
        
        # alternative using openCV results results in closed chains (might be better), but a few chains are missing
        # hierarchy,contours = cv2.findContours(pixel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # for h in hierarchy:
        #     h2 = h.reshape((-1,2))
        #     h3 = [list(xy)[::-1] for xy in h2]
        # if len(h3)>3:
        #     chains3 += [ h3 ]
        
        
        while True:
            points = np.argwhere(pixel!=0)
            if len(points)==0: break
            x,y = points[0] # set starting point
            done = False
            subchain = []
            while not done:
                subchain += [[x,y]]
                pixel[x,y] = 0
                done = True
                for dx,dy in [(+1,0),(-1,0),(+1,-1),(-1,+1),(0,-1,),(0,+1),(-1,-1),(+1,+1)]:
                    if x+dx>=0 and x+dx<pixel.shape[0] and y+dy>=0 and y+dy<pixel.shape[1]: # prüfen ob im Bild drin
                        if pixel[x+dx,y+dy]>0: # check for pixel here
                            x,y = x+dx, y+dy # if yes, jump here
                            done = False # tell the middle loop that the chain is not finished
                            break # break inner loop
            if len(subchain)>half_tile//2:
                chains += [subchain]
   
    return chains



def chains_and_angles(img_edges, half_tile, plot=[]):

    # for each pixel get distance to closest edge
    distances = morphology.distance_transform_edt(img_edges==0,)

    # tiles will be placed centered along guidelines (closed lines)
    """     tile
           xxxxxx
           xxxxxx
    ---------------------- guideline
           xxxxxx
           xxxxxx
    """
    w,h = img_edges.shape[0],img_edges.shape[1]
    guidelines = np.zeros((w, h), dtype=np.uint8)
    mask = ( (distances.astype(int)+half_tile) % (2*half_tile)==0)
    guidelines[mask] = 1
    # break into chains and order the points
    t0 = time.time()
    chains = pixellines_to_ordered_points(guidelines, half_tile)
    print ('Pixel guidelines to chains with sorted points:', f'{time.time()-t0:.1f}s')

    # use distances to calculate gradients => rotation of tiles when placed later
    t0 = time.time()
    gradient = np.zeros((w,h))
    for x in range(1,w-1):
        for y in range(1,h-1):
            numerator = distances[x,y+1]-distances[x,y-1]
            denominator = distances[x+1,y]-distances[x-1,y]
            gradient[x,y] = np.arctan2(numerator, denominator)
    angles_0to180 = (gradient*180/np.pi+180) % 180
    print ('Calculation of angle matrix:', f'{time.time()-t0:.1f}s')  
    # Remark: it would be enough to calculate only x,y inside the chain => faster
    # interim_stages = dict(distances=distances, guidelines=guidelines, chains=chains,
    #                       gradient=gradient, angles_0to180=angles_0to180)
    
    if 'distances' in plot: plotting.plot_image(distances, title='distances')
    if 'guidelines' in plot: plotting.plot_image(guidelines, inverted=True, title='guidelines')
    if 'gradient' in plot: plotting.plot_image(gradient, title='gradients')
    if 'angles_0to180' in plot: plotting.plot_image(angles_0to180)
    
    return chains, angles_0to180#, interim_stages



def chains_into_gaps(polygons, h, w, half_tile, CHAIN_SPACING, plot=[]):
    # get area which are already occupied
    img_chains = np.zeros((h, w), dtype=np.uint8)
    for p in polygons:
        y,x = p.exterior.coords.xy
        rr, cc = draw.polygon(x, y, shape=img_chains.shape)
        img_chains[rr, cc] = 1
    distance_to_tile = morphology.distance_transform_edt(img_chains==0)
    d = distance_to_tile.astype(int)
    
    # define new guidelines
    chain_spacing = int(round(half_tile*CHAIN_SPACING)) 
    if chain_spacing <= 1: # would select EVERY pixel inside gap
        chain_spacing = 2
    # first condition (d==1) => chains around all (even the smallest) gap borders
    # (set e.g. d==2 for faster calculations)
    # second condition (...) => more chains inside larger gaps
    mask = (d==1) | ( (d%chain_spacing==0) & (d>0) )
    
    guidelines2 = np.zeros((h, w), dtype=np.uint8)
    guidelines2[mask] = 1
    chains2 = pixellines_to_ordered_points(guidelines2, half_tile)
    
    if 'used_up_space' in plot: plotting.plot_image(img_chains, title='gaps')
    if 'distance_to_tile' in plot: plotting.plot_image(distance_to_tile, inverted=True)
    if 'filler_guidelines' in plot: plotting.plot_image(guidelines2, inverted=True, title='new guidelines')
        
    return chains2




if __name__ == '__main__':
    img = sk.data.coffee()
    import edges
    img_edges = edges.edges_diblasi(img)
    img_edges = edges.edges_hed(img, gauss=0)
    chains, angles_0to180 = chains_and_angles(img_edges, half_tile=10)








