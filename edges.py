#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage as sk
from skimage.io import imread
from skimage import filters, transform, color, exposure, morphology
import plotting
from pathlib import Path
import cv2 as cv


def load_image(fname, width=900, plot=[]):
    if fname:
        img0 = imread(fname)
    else:
        img0 = sk.data.coffee() # coffee (example image)
    
    # Ensure image is RGB (for consistency)
    if len(img0.shape) < 3:
        img0 = color.gray2rgb(img0) 

    # Resize to same image width => tile size has always similar effect
    if width is not None:
        factor = width / img0.shape[1]
        img0 = transform.resize(img0, (int(img0.shape[0] * factor), int(img0.shape[1] * factor)), anti_aliasing=True)
    img0 = (img0 * 255).astype(int)
    if 'original' in plot: plotting.plot_image(img0)
    print(f'Size of input image: {img0.shape[0]}px * {img0.shape[1]}px')
    
    return img0


def edges_diblasi(img, gauss=5, details=1, plot=[]):
    # Convert RGB to gray ("Luminance channel" in Di Blasi)
    img_gray = color.rgb2gray(img)
    
    # Equalize histogram
    img_eq = exposure.equalize_hist(img_gray)
    
    # Soften image
    img_gauss = filters.gaussian(img_eq, sigma=gauss)
    
    # Segment bright areas to blobs
    variance = img_gauss.std() ** 2
    img_seg = np.ones((img.shape[0], img.shape[1]))
    threshold = variance / 4 * 2 * details
    img_seg[np.abs(img_gauss - img_gauss.mean()) > threshold] = 0
    
    # Find edges
    img_edge = filters.laplace(img_seg, ksize=3)
    img_edge[img_edge != 0] = 1
    
    if 'edges' in plot:
        plotting.plot_image(img_edge, inverted=True, title='Di Blasi')
    
    return img_edge


def hed_edges(image):
    class CropLayer(object):
        def __init__(self, params, blobs):
            self.xstart = 0
            self.xend = 0
            self.ystart = 0
            self.yend = 0

        def getMemoryShapes(self, inputs):
            inputShape, targetShape = inputs[0], inputs[1]
            batchSize, numChannels = inputShape[0], inputShape[1]
            height, width = targetShape[2], targetShape[3]
            self.ystart = int((inputShape[2] - targetShape[2]) / 2)
            self.xstart = int((inputShape[3] - targetShape[3]) / 2)
            self.yend = self.ystart + height
            self.xend = self.xstart + width
            return [[batchSize, numChannels, height, width]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

    # Load the pretrained model
    script_path = Path(__file__).parent.absolute()
    hed_path = Path.joinpath(script_path, 'HED')
    net = cv.dnn.readNetFromCaffe(str(hed_path / 'deploy.prototxt'),
                                  str(hed_path / 'hed_pretrained_bsds.caffemodel'))
    cv.dnn_registerLayer('Crop', CropLayer)

    image = cv.resize(image, (image.shape[1], image.shape[0]))
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[1], image.shape[0]), # w,h
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    cv.dnn_unregisterLayer('Crop')  # get rid of issues when run in a loop
    out = out[0, 0]
    return out


def edges_hed(img, gauss=None, plot=[]):
    if gauss is not None:
        # Apply Gaussian filter without 'multichannel' argument
        img = filters.gaussian(img, sigma=16)  # Adjust sigma and other parameters as needed
    
    img = img / np.amax(img) * 255
    img = img.astype(np.uint8)    
    
    hed_matrix = hed_edges(img)
    
    # Convert gray to binary
    hed_seg = np.ones((hed_matrix.shape[0], hed_matrix.shape[1]))
    hed_seg[hed_matrix < 0.5] = 0
    
    # Skeletonize to get inner lines
    img_edges = morphology.skeletonize(hed_seg).astype(int)

    # Option to make plot lines thicker:
    # from skimage.morphology import square, dilation
    # img_edges = dilation(img_edges, square(3))
    
    if 'edges' in plot:
        plotting.plot_image(img_edges, inverted=True, title='HED')
    
    return img_edges


if __name__ == '__main__':
    img = sk.data.coffee()
    # img_edges = edges_diblasi(img)
    img_edges = edges_hed(img, gauss=0)
