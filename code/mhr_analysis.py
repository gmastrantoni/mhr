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

#%% General Functions
def open_raster(raster_file_path):
    with rasterio.open(raster_file_path) as src:
        #get transform
        transform = src.meta['transform']
        #create 2darray from band 1
        array = src.read(1)
        # print("Array shape: ", array.shape)
        
        # print('src type: ', type(src))
        # get raster profile
        # print('no data value: ', src.nodata)
        kwds = src.profile

        return array, kwds, transform

def save_raster(raster_array, name, kwds):
    with rasterio.open(os.path.join(name+".tif"), 'w', **kwds) as dst:
        dst.write(raster_array, indexes=1)

def drop_columns(df, cols):
    df_clean = df.drop(columns=cols)
    return df_clean


#%% HAZARD SCORE
print('Computing Hazard Scores...')
### Functions for Hazard Analysis ###
def zs(polygons, stats, raster_path, prefix='', nodatavalue=-9999, categorical=False, category_map=None, buffering=False, drop_zeros=True):
    """ Function for zonal statistics of raster array on polygon vectors."""
    # OPEN RASTER FILE (.tif)
    raster_array, kwds, transform = open_raster(raster_path)
    # replace nodata value with 0
    raster_array[raster_array == nodatavalue] = 0    
    # PERFORM ZONAL STATS
    if buffering==True:
        # Buffering
        polygons_buffered = polygons.buffer(20, cap_style=3, join_style=2)
        poly_zs = zonal_stats(polygons_buffered, raster_array, affine=transform, stats=stats, prefix=prefix, categorical=categorical, category_map=category_map)
        poly_zs_df = pd.DataFrame(poly_zs); del poly_zs
        # INSERIRE NP.CEIL FUNCTION TO ROUND UP VALUES !!
        polygons_zs = polygons.merge(poly_zs_df, left_index=True, right_index=True)
    else:
        poly_zs = zonal_stats(polygons, raster_array, affine=transform, stats=stats, prefix=prefix, categorical=categorical, category_map=category_map)
        poly_zs_df = pd.DataFrame(poly_zs); del poly_zs
        polygons_zs = polygons.merge(poly_zs_df, left_index=True, right_index=True)
    # Drop polygons with 0 pixel
    if drop_zeros == True:
        pn = prefix+'count'
        zeros = (polygons_zs[pn] == 0)
        polygons_zs_clear = polygons_zs.drop(polygons_zs[zeros].index)
        return polygons_zs_clear
    else: return polygons_zs

def exposure(polygons, col_name, min_exposure_value):
    polygons['Exposure'] = polygons[col_name].apply(lambda x: True if x>=min_exposure_value else False)
    return polygons

#### ----------- CALLING FUNCTIONS  -------- ####
stats = ['count', 'std', 'max']
# LANDSLIDE
print('Landslide Hazard...')
filepath_ls= "SUSCEPTIBILITIES/landslide_pre2014_32633.tif"
builds_zs_landslide = zs(builds_w_all, stats, filepath_ls, prefix='ls_', buffering=True)
# Drop repeated columns
cols = builds_w_all.columns[1:-1]
builds_zs_landslide = drop_columns(builds_zs_landslide, cols)
# Exposure Analysis
exp_builds_landslide = exposure(builds_zs_landslide, 'ls_max', min_exposure_value=2)

# SUBSIDENCE
print('Subsidence Hazard...')
filepath_subs= "SUSCEPTIBILITIES\susc_subdidenza_32633_esteso_rescaled_new.tif"
builds_zs_subsidence = zs(builds_w_all, stats, filepath_subs, prefix='subs_', buffering=False)
# Drop Repeated columns
builds_zs_subsidence = drop_columns(builds_zs_subsidence, cols)
# Exposure Analysis
    # Round Up values
builds_zs_subsidence['subs_max'] = np.ceil(builds_zs_subsidence['subs_max'])
exp_builds_subsidence = exposure(builds_zs_subsidence, 'subs_max', min_exposure_value=2)

# SINKHOLE
print('Sinkhole Hazard...')
filepath_shsi= "SUSCEPTIBILITIES/SHSI_32633.tif"
builds_zs_sinkhole = zs(builds_w_all, stats, filepath_shsi, prefix='shsi_', buffering=False)
# Drop Repeated Cols
builds_zs_sinkhole = drop_columns(builds_zs_sinkhole, cols)
# Exposure Analysis
    # Round Up values
builds_zs_sinkhole['shsi_max'] = np.ceil(builds_zs_sinkhole['shsi_max'])
exp_builds_sinkhole = exposure(builds_zs_sinkhole, 'shsi_max', min_exposure_value=2)

### Save gdf
filepath= 'SUSCEPTIBILITIES/ZONAL_STATS/new'
exp_builds_landslide.to_file(os.path.join(filepath, 'build_exposure_landslide_new.gpkg'), driver='GPKG')
exp_builds_subsidence.to_file(os.path.join(filepath, 'build_exposure_subsidence_new.gpkg'), driver='GPKG')
exp_builds_sinkhole.to_file(os.path.join(filepath, 'build_exposure_sinkhole_new.gpkg'), driver='GPKG')


#%% ACTIVITY SCORE
print('Computing Activity Scores...')

