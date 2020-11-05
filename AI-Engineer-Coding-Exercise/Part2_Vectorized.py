# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:57:23 2020

@author: tchat
"""

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

ref = np.load('data/ref_spectra_standardized.npy')

ds = gdal.Open("data/ortho/swir_ortho_standardized.tif")
data = np.array(ds.ReadAsArray())

dataT = data.transpose()

output = []
for i in range(502):
    dots = np.dot(dataT[0], ref.T)
    l2norms = np.sqrt(((dataT[0]**2).sum(1)[:,None])*((ref**2).sum(1)))
    Cdist = 1 - (dots/l2norms)
    output.append(Cdist)
    
Result = np.array(output)  
Result = Result.transpose(1,0,2)  