#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 06:39:42 2021

@author: nn
"""

from pathlib import Path
import numpy as np
from skimage import io
from skimage import draw


def colors_from_original(polygons, original_image, method='average'):
    colors = []
    for j,p in enumerate(polygons): 

        if method == 'point':
            x,y = np.array(p.representative_point().coords[0]).astype(int)
            color = original_image[y,x,:]/255
        
        elif method == 'average':
            xx,yy = p.exterior.xy
            x_list, y_list = draw.polygon(xx, yy)
            if len(x_list)>1 and len(y_list)>1: 
                img_cut = original_image[min(y_list):max(y_list)+1, min(x_list):max(x_list)+1, :]
                #https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
                average = img_cut.mean(axis=0).mean(axis=0)
                color = average/255
            else:
                color = original_image[int(yy[0]),int(xx[0]),:]/255

        else:
            raise ValueError('Parameter not understood.')
        
        colors += [color]
        
    return colors



def modify_colors(colors, variant, colors_collection=[]):
    def nearest_color( subjects, query ):
        # https://stackoverflow.com/questions/34366981/python-pil-finding-nearest-color-rounding-colors
        return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )
    #nearest_color( ((1, 1, 1, "white"), (1, 0, 0, "red"),), (64/255,0,0) ) # example
    new_colors = []
    for c in colors:
        if variant == 'monochrome':
            c_new = nearest_color(((1,1,1),(0,0,0)), c) # monochrom
        elif variant == 'grayscale':
            c_new = str(.2989 * c[0] + 0.5870*c[1] + 0.1140*c[2]) # matplotlib excepts grayscale be strings
        elif variant == 'polychrome':
            n = 9
            some_gray = [(g/n,g/n,g/n) for g in range(n+1)]
            c_new = nearest_color(some_gray, c) # monochrom            
        elif variant == 'source':
            c_new = nearest_color(colors_collection/255, c)
        else:
            raise ValueError('Parameter not understood.')
        new_colors += [c_new]
    return new_colors


def load_colors():
    script_path = Path(__file__).parent.absolute()
    collection_path = Path.joinpath(script_path, 'color_collections')
    color_dict = {}
    for fname in collection_path.glob('*.npy'):
        color_dict[fname.stem] = np.load(fname)
    return color_dict


def extract_colors(input_fname):
    # color quantization of real mosaic images 
    # => saved as *.npy into subpath "color_collection"
    # based on https://stackoverflow.com/questions/48222977/python-converting-an-image-to-use-less-colors
    from sklearn.cluster import KMeans
    n_colors = 10
    sample_size = 200000 # reduce sample size to speed up KMeans
    original = io.imread(input_fname)
    original = original[:,:,:3] # drop alpha channel if there is one
    arr = original.reshape((-1, 3))
    random_indices = np.random.choice(arr.shape[0],size=sample_size,replace=True)
    arr = arr[random_indices,:]
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    centers = kmeans.cluster_centers_
    script_path = Path(__file__).parent.absolute()
    out_path = Path.joinpath(script_path, 'color_collections')
    np.save(out_path / input_fname.name, centers)




if __name__ == '__main__':
    
    data_path = ''
    fnames = []
    for fname in fnames:
        extract_colors(Path(data_path) / fname)
































