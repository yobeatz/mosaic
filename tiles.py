#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import random
from shapely.geometry import LineString, Polygon, MultiPoint
from shapely import affinity
import plotting


def fit_in_polygon(p, nearby_polygons):
    # Remove parts from polygon which overlap with existing ones:
    for p_there in nearby_polygons: 
        p = p.difference(p_there)
    # only keep largest part if polygon consists of multiple fragments:
    if p.geom_type=='MultiPolygon':
        i_largest = np.argmax([p_i.area for p_i in p])
        p = p[i_largest]
    # remove pathologic polygons with holes (rare event):
    if p.type not in ['MultiLineString','LineString', 'GeometryCollection']:
        if p.interiors: # check for attribute interiors if accessible
            p = Polygon(list(p.exterior.coords))
    return p


def place_tiles_along_chains(chains, angles_0to180, half_tile, RAND_SIZE, MAX_ANGLE, A0, plot=[]):
    # construct tiles along ductus chain
    RAND_EXTRA = int(round(half_tile*RAND_SIZE))  
    
    polygons = []
    t0 = time.time()
    delta_i = int(half_tile*2) # width of standard tile (i.e. on straight lines)
    for ik, chain in enumerate(chains):
    
        # consider existing polygons next to the new lane (reason: speed)
        search_area = LineString(np.array(chain)[:,::-1]).buffer(2.1*half_tile)
        preselected_nearby_polygons = [poly for poly in polygons if poly.intersects(search_area)]
    
        for i in range(len(chain)):
            y,x = chain[i]
            winkel = angles_0to180[y,x]
            
            if i == 0: # at the beginning save the first side of the future polygon
                i_start = i
                rand_i = random.randint(-RAND_EXTRA,+RAND_EXTRA) # a<=x<=b
                winkel_start = winkel
                line_start = LineString([(x,y-half_tile),(x,y+half_tile)])
                line_start = affinity.rotate(line_start, -winkel_start)
            
            # Draw polygon as soon as one of the three conditions is fullfilled:
            draw_polygon = False
            # 1. end of chain is reached
            if i==len(chain)-1: 
                draw_polygon = True
            else:
                y_next, x_next = chain[i+1]
                winkel_next = angles_0to180[y_next,x_next]
                winkeldelta = winkel_next-winkel_start
                winkeldelta = min( 180-abs(winkeldelta), abs(winkeldelta))     
                # 2. with the NEXT point a large angle would be reached => draw now
                if winkeldelta > MAX_ANGLE:
                    draw_polygon = True
                # 3. goal width is reached
                if i-i_start == delta_i+rand_i:
                    draw_polygon = True
                    
            if draw_polygon:
                
                line = LineString([(x,y-half_tile),(x,y+half_tile)])
                line = affinity.rotate(line, -winkel)
    
                # construct new tile
                p = MultiPoint([line_start.coords[0], line_start.coords[1], line.coords[0], line.coords[1]])
                p = p.convex_hull
                
                line_start = line
                winkel_start = winkel
    
                # do not draw very thin polygon, but set as new starting point (line_start) to skip critical area
                if i-i_start <= 2: 
                    i_start = i
                    continue
                i_start = i
                rand_i = random.randint(-RAND_EXTRA,+RAND_EXTRA) # a<=x<=b
    
                # cut off areas that overlap with already existing tiles
                nearby_polygons = [poly for poly in preselected_nearby_polygons if p.disjoint(poly)==False]
                p = fit_in_polygon(p, nearby_polygons)
                
                # Sort out small tiles
                if p.area >= 0.08*A0 and p.geom_type=='Polygon' and p.is_valid: 
                    polygons += [p]
                    preselected_nearby_polygons += [p]
            
    print (f'Placed {len(polygons)} tiles along guidelines', f'{time.time()-t0:.1f}s') 
    
    if 'polygons_chains' in plot: 
        plotting.draw_tiles(polygons, None, h=0,w=0, background_brightness=0.2,
                            return_svg=False, chains=chains, axis_off=True)
   
    return polygons





def place_tiles_into_gaps(polygons, filler_chains, half_tile, A0, plot=[]):
    # fill spaces which are still empty after the main construction step
    t0 = time.time()
    counter = 0
    for chain in filler_chains:
        # Speed up:
        chain_as_line = LineString(np.array(chain)[:,::-1]).buffer(2.1*half_tile) # ::-1 weil x und y vertauscht werden muss
        preselected_nearby_polygons = [poly for poly in polygons if poly.intersects(chain_as_line)]
        # Sicherstellen, dass am Ende der Kette nichts verschenkt wird
        index_list = list(range(0, len(chain), half_tile*2))
        last_i = len(chain)-1
        min_delta = 3
        if index_list[-1] != last_i and last_i-index_list[-1]>=min_delta:
            index_list += [last_i]
        for i in index_list:
            y,x = chain[i]
            p = Polygon([[x-half_tile, y+half_tile], [x+half_tile, y+half_tile],
                         [x+half_tile, y-half_tile], [x-half_tile, y-half_tile]])
            # fit in polygon (concave ones are okay for now)
            p_buff = p.buffer(0.1)
            nearby_polygons = [poly for poly in preselected_nearby_polygons if p_buff.intersects(poly)] 
            for p_vorhanden in nearby_polygons:
                try:
                    p = p.difference(p_vorhanden) # => remove overlap
                except:
                    p = p.difference(p_vorhanden.buffer(0.1)) # => remove overlap
            # keep only largest fragment if more than one exists
            if p.geom_type=='MultiPolygon':
                i_largest = np.argmax([p_i.area for p_i in p])
                p = p[i_largest]
            if p.area >= 0.05*A0 and p.geom_type=='Polygon': # sort out very small tiles
                polygons += [p]
                preselected_nearby_polygons += [p]
                counter += 1
    if 'polygons_filler' in plot: 
        plotting.draw_tiles(polygons, None, h=0,w=0, background_brightness=0.2,
                            return_svg=False, chains=filler_chains, axis_off=True)                
    print (f'Added {counter} tiles into gaps:', f'{time.time()-t0:.1f}s')
    return polygons


def cut_tiles_outside_frame(polygons, half_tile, w, h, plot=[]):
    # remove parts of tiles which are outside of the actual image
    t0 = time.time()
    A0 = (2*half_tile)**2
    outer = Polygon([ (-3*half_tile,-3*half_tile),(h+3*half_tile,-3*half_tile),
                          (h+3*half_tile,w+3*half_tile),(-3*half_tile,w+3*half_tile) ],
                        holes=[[ (1,1),(h-1,1),(h-1,w-1),(1,w-1) ],] 
                        )
    polygons_cut = []
    counter = 0
    for j,p in enumerate(polygons):
        y,x = list(p.representative_point().coords)[0]
        if y<4*half_tile or y>h-4*half_tile or x<4*half_tile or x>w-4*half_tile:
            p = p.difference(outer) # => if outside image borders
            counter += 1
        if p.area >= 0.05*A0 and p.geom_type=='Polygon':
            polygons_cut += [p]
    if 'polygons_cut' in plot: 
        plotting.draw_tiles(polygons_cut, None, h=0,w=0, background_brightness=0.2,
                            return_svg=False, chains=None, axis_off=True)              
    print (f'Up to {counter} tiles beyond image borders were cut', f'{time.time()-t0:.1f}s') 
    return polygons_cut



def irregular_shrink(polygons, half_tile):
    polygons_shrinked = []
    for p in polygons:
        p = affinity.scale(p, xfact=random.uniform(0.85, 1),
                              yfact=random.uniform(0.85, 1))
        p = p.buffer(-0.03*half_tile)
        #p = affinity.rotate(p, random.uniform(-5,5))
        #p = affinity.skew(p, random.uniform(-5,5),random.uniform(-5,5))
        polygons_shrinked += [p]
    return polygons_shrinked



def repair_tiles(polygons):
    # remove or correct strange polygons
    polygons_new = []
    for p in polygons:
        if p.type == 'MultiPolygon':
            for pp in p:
                polygons_new += [pp]
        else:
            polygons_new += [p]
    
    polygons_new2 = []
    for p in polygons_new:
        if p.exterior.type == 'LinearRing':
            polygons_new2 += [p]
    
    return polygons_new2


def reduce_edge_count(polygons, half_tile, tol=20):
    polygons_new = []
    for p in polygons:
        p = p.simplify(tolerance=half_tile/tol)
        polygons_new += [p]
    return polygons_new


def drop_small_tiles(polygons, A0, threshold=0.03):
    polygons_new = []
    counter = 0
    for p in polygons:
        if p.area > threshold*A0:
            polygons_new += [p]
        else:
            counter += 1
    print (f'Dropped {counter} small tiles ')
    return polygons_new





