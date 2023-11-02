@author: giandomenico mastrantoni - giandomenico.mastrantoni@uniroma1.it
#%% import libraries
import random
import math

import rasterio
from rasterio.plot import show
from rasterstats import zonal_stats

import geopandas as gpd
import mapclassify

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import contextily as cx

#%% Set main folder path
os.chdir(".../data") # insert your folder path here.

#%% READ PRE-PROCESSED INPUT DATA
print('Reading Input Data...')
# Area Of Interest
AOI = gpd.read_file("AOI.gpkg", ignore_fields=['location', 'id', 'objectid', 'cod_pro','cod_istat', 'pro_com', 'nome', 'shape_leng', 'shape_area'])
# Italian Real Estate Market Observatory (Osservatorio Mercato Immobiliare - Agenzia delle Entrate)
OMI = gpd.read_file("ASSETS/pre-processed/OMI_AOI.gpkg")
# Census Tracts
cs = gpd.read_file("ASSETS/sez_censuarie_2011_Roma_32633.gpkg",
                       ignore_fields=['pro_com', 'cod_stagno','cod_fiume', 
                                      'cod_lago', 'cod_laguna', 'cod_val_p', 'cod_zona_c',
                                      'cod_is_amm', 'cod_is_lac', 'cod_is_mar', 'cod_area_s', 'cod_mont_d',
                                      'loc2011', 'cod_loc', 'tipo_loc', 'com_asc', 'cod_asc', 'ace',
                                      'shape_leng'
                                      ]
                       )

# Buildings from DBSN Esercito, already pre-processed (output from data_preparation.py)
builds_w_all = gpd.read_file("ASSETS/pre-processed/builds_w_all.gpkg")



