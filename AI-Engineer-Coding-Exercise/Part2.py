# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:31:20 2020

@author: tchat
"""

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cs


ref = np.load('data/ref_spectra_standardized.npy')

ds = gdal.Open("data/ortho/swir_ortho_standardized.tif")
data = np.array(ds.ReadAsArray())

dataT = data.transpose()

m,n = dataT.shape[:2]
output = cs(dataT.reshape(m*n,-1), ref).reshape(m,n,-1)
out = output.transpose(1,0,2)