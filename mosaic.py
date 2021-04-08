#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an pixel image into a mosaic constructed by polygons
=======================================================================
Author: Johannes Beetz
Based on algorithm described in "Artificial Mosaics (2005)" by Di Blasi
"""

import time
import random
import edges, guides, tiles, convex, coloring, plotting

# Select filename of input image
fname = '' # let empty for test image


# Parameters
half_tile = 12 # 4...30 => half size of mosaic tile
GAUSS = 3 # 0...8 => blurs image before edge detection (check "edges" image for a good value)
EDGE_DETECTION = 'HED' # HED or DiBlasi
WITH_FRAME = True # default is True => control about guidelines along image borders
RAND_SIZE = 0.3 # portion of tile size which is added or removed randomly during construction
MAX_ANGLE = 40 # 30...75 => max construction angle for tiles along roundings
GAP_CHAIN_SPACING = 0.5 # 0.4 to 1.0 => spacing of gap filler chains
MAKE_CONVEX = True # default is True => break concave into more realistic polygons
COLOR_SCHEMA = ['nilotic',] # leave empty to plot all available or choose from 'wise_men',
                  #  'fish', 'cave_canem', 'nilotic', 'rooster', 'carpe_diem', 'Hyena'

# choose which image to plot
plot_list = [
    #'original',
    'edges', # CAN BE HELPFUL FOR ADJUSTING GAUSS
    #'distances',
    #'guidelines',
    #'gradient',
    #'angles_0to180',
    #'polygons_chains',
    #'used_up_space',
    #'distance_to_tile',
    #'filler_guidelines',
    #'polygons_filler',
    #'polygons_cut',
    'final', # <== MOST IMPORTANT
    'final_recolored', # <== OR THIS
    #'statistics',
    ]


# Load image
t_start = time.time()
random.seed(0)
img0 = edges.load_image(fname, plot=plot_list)
h,w = img0.shape[0],img0.shape[1]
A0 = (2*half_tile)**2 # area of tile when placed along straight guideline
print (f'Estimated number of tiles: {2*w*h/A0:.0f}') # factor 2 since tiles can be smaller than default size

# Find edges of image objects
if EDGE_DETECTION == 'HED':
    img_edges = edges.edges_hed(img0, gauss=GAUSS, plot=plot_list)
elif EDGE_DETECTION == 'DiBlasi':
    img_edges = edges.edges_diblasi(img0, gauss=GAUSS, details=4, plot=plot_list)
else:
    raise ValueError('Parameter for edge detection mode not understood.')


if WITH_FRAME: 
    img_edges[0,:]=1; img_edges[-1,:]=1; img_edges[:,0]=1; img_edges[:,-1]=1


# place tiles along chains
chains, angles_0to180 = guides.chains_and_angles(img_edges, half_tile=half_tile, plot=plot_list)
polygons_chains = tiles.place_tiles_along_chains(chains, angles_0to180, half_tile, RAND_SIZE, MAX_ANGLE, A0, plot=plot_list)

# find gaps and put more tiles inside
filler_chains = guides.chains_into_gaps(polygons_chains, h, w, half_tile, GAP_CHAIN_SPACING, plot=plot_list)
polygons_all = tiles.place_tiles_into_gaps(polygons_chains, filler_chains, half_tile, A0, plot=plot_list)

# remove parts of tiles which reach outside of image frame
polygons_all = tiles.cut_tiles_outside_frame(polygons_all, half_tile, img0.shape[0],img0.shape[1], plot=plot_list)

# convert concave polygons to convex (i.e. more realistic) ones
polygons_convex = convex.make_convex(polygons_all, half_tile, A0) if MAKE_CONVEX else polygons_all

# make polygons smaller, remove or correct strange polygons, simplify and drop very small polygons
polygons_post = tiles.irregular_shrink(polygons_convex, half_tile)
polygons_post = tiles.repair_tiles(polygons_post) 
polygons_post = tiles.reduce_edge_count(polygons_post, half_tile)
polygons_post = tiles.drop_small_tiles(polygons_post, A0)

if 'final' in plot_list:
    # copy colors from original image
    colors = coloring.colors_from_original(polygons_post, img0, method='average')
    
    t0 = time.time()
    svg = plotting.draw_tiles(polygons_post, colors, h,w, background_brightness=0.2, return_svg=False, chains=None)
    if svg:
        with open("output.svg", "w") as fn:
            fn.write(svg)
    print ('Final plot / saving:', f'{time.time()-t0:.1f}s') 

if 'final_recolored' in plot_list:
    color_dict = coloring.load_colors()
    keys = color_dict.keys() if not COLOR_SCHEMA else COLOR_SCHEMA
    for key in keys:
        new_colors = coloring.modify_colors(colors, 'source', color_dict[key])
        title = key if not COLOR_SCHEMA else ''
        plotting.draw_tiles(polygons_post, new_colors, h, w, background_brightness=0.2,
                            return_svg=None, chains=None, title=title)

if 'statistics' in plot_list:
    plotting.statistics(polygons_post)

print (f'Total calculation time: {time.strftime("%M min %S s", time.gmtime((time.time()-t_start)))}' ) # sek->min:sek
print ('Final number of tiles:', len(polygons_post))
























