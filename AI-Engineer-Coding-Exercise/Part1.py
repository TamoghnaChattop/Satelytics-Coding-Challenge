# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:30:52 2020

@author: tchat
"""

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

ds = gdal.Open("data/ortho/swir_ortho_standardized.tif")
data = np.array(ds.ReadAsArray())

faculty_spectrum = np.load('data/facility_spectrum_standardized.npy')

dataT = data.transpose()

result = [[] for i in range(653)]

for i in range(502):
    for j in range(653):
        dot_prod = np.dot(faculty_spectrum, dataT[i][j].transpose())
        norm_prod = np.linalg.norm(faculty_spectrum)*np.linalg.norm(dataT[i][j])
        cos_sim = dot_prod/norm_prod
        result[j].append(cos_sim)

output = np.array(result)

op_plot = plt.imshow(output)
plt.savefig('Answers/Cosine_Similarity_Part1.png')

threshold = 0.995
output2 = np.zeros((653,502))

for i in range(653):
    for j in range(502):
        if output[i][j]>threshold:
            output2[i][j] = 1
        else:
            output2[i][j] = 0
            
mask = plt.imshow(output2)
plt.savefig('Answers/Mask_Part1.png')