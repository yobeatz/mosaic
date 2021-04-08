#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from shapely.geometry import LineString, Polygon, MultiPoint#,Point
from shapely import affinity


def fit_in_polygon(p, nearby_polygons):
    # Remove parts from polygon which overlap with existing ones:
    for p_vorhanden in nearby_polygons: 
        p = p.difference(p_vorhanden)
    # only keep largest part if polygon consists of multiple fragments:
    if p.geom_type=='MultiPolygon':
        i_largest = np.argmax([p_i.area for p_i in p])
        p = p[i_largest]
    # remove pathologic polygons with holes (rare event):
    if p.type not in ['MultiLineString','LineString', 'GeometryCollection']:
        if p.interiors: # check for attribute interiors if accessible
            p = Polygon(list(p.exterior.coords))
    return p


def my_simplify(p, accepted_loss=0.05): # concave->convex
    # Prüfen, ob das Polygon durch Weglassen einer Ecke (oder von mehr als einer), 
    # immer noch ähnlich aussieht (d.h. Flächeninhalt nur minimal kleiner, keinesfalls größer!)
    # => besonders nützlich, um "fiese Spitzen" bei konkaven Polygonen zu entfernen
    while True:
        ecken = list(p.exterior.coords)[:-1] # remove last coordinate (is only repeated from first)
        A0 = Polygon(ecken).area
        erfolg = False
        for i in range(len(ecken)):
            ecken_neu = ecken[:i]+ecken[i+1:] # remove a corner
            if len(ecken)<=3: break # must be at leat a triangle
            p_neu = Polygon(ecken_neu)
            if p_neu.area <= A0 and A0-p_neu.area < accepted_loss*A0 and p_neu.area>0.05*A0 and p_neu.is_valid:
                p = p_neu # besseres Polygon
                erfolg = True
                break
        if len(ecken)<3: erfolg=False # must be at leat a triangle
        if p.geom_type!='Polygon': erfolg=False
        if erfolg == False: break
    return p

def is_convex(p):
    if p.convex_hull.area > 1.01 * p.area:
        return False
    else:
        return True

def simple_concave_zu_convex(p, half_tile, A0, richtung=-1):
    # Split off polygon corners that lie "inside" (i.e. concave) 
    concave_list = [p] # bad ones
    convex_list = [] # goal
    success = True
    counter = 0
    while len(concave_list)>0:
        counter += 1
        p = concave_list.pop()
        p_points = MultiPoint(p.exterior.coords)
        concave_points = [i for i,point in enumerate(p_points) if p.convex_hull.contains(point)]
        # does not find points when polygon has hole (i.e. has interiors)
        if len(concave_points) == 0:
            print ('Could not convert tile to convex. Should not happen :-(')
            return False, []
        i_krit = concave_points[0]
        xa,ya = p_points[i_krit].coords[0]
        xb,yb = p_points[i_krit+richtung].coords[0] # -1 or +1 can be used
        angle_of_cut_line = np.arctan2(xa-xb, ya-yb)*180/np.pi
        cut_line = LineString([(xa,ya-half_tile*4),(xa,yb+half_tile*4)])
        cut_line = affinity.rotate(cut_line, -angle_of_cut_line, origin=p_points[i_krit])
        try:
            pp = p.difference(cut_line.buffer(0.2)) # remark: shapely.split() is not useful
        except: # rarely: TopologicalError: This operation could not be performed. Reason: unknown
            success = False
            break
        if pp.geom_type != 'MultiPolygon':
            success = False
            break
        if counter>5:
            success = False
            break
        for ppi in pp:
            if ppi.is_valid == False or ppi.area<0.05*A0:
                continue
            if not is_convex(ppi):
                convex_list += [ppi]
            else:
                convex_list += [ppi]
    return success, convex_list



def make_convex(polygons, half_tile, A0):
    t0 = time.time()
    still_concave = []
    polygons_convex = []
    counter = 0
    for j,p in enumerate(polygons):
        if is_convex(p):
            polygons_convex += [p] # ideal case
            counter += 1
        else:
            p = my_simplify(p)
            if is_convex(p):
                polygons_convex += [p]
            else:
                success, convex_list = simple_concave_zu_convex(p, half_tile, A0, richtung=-1)
                if not success:
                    # second chance with other cutting direction:
                    success, convex_list = simple_concave_zu_convex(p.buffer(0.1), half_tile, A0, richtung=+1)
                # and again, but buffered this time
                if not success:
                    success, convex_list = simple_concave_zu_convex(p.buffer(0.5), half_tile, A0, richtung=+1)
                if not success:
                    success, convex_list = simple_concave_zu_convex(p.buffer(0.5), half_tile, A0, richtung=-1)
                if success:
                    for kp in convex_list:
                        polygons_convex += [kp]

                else:
                    accepted_loss = 0.05 # default
                    while is_convex(p)==False and accepted_loss<0.8:
                        accepted_loss += 0.05
                        p = my_simplify(p, accepted_loss)
                    if is_convex(p):
                        polygons_convex += [p]
                    else:
                        still_concave += [p]

    if still_concave:
        print (f'! {len(still_concave)} tiles are still concave')    
    print (f'{len(polygons)-counter} tiles converted to convex', f'{time.time()-t0:.1f}s')    
    return polygons_convex









