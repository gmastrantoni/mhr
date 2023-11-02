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

# INSAR RESULTS of Buildings' Displacement from PS-Toolbox.
filepath_Uz = "PS-INSAR/DF_Results/builds_displ_Uz_32633.tif" #vertical component.
filepath_Ux = "PS-INSAR/DF_Results/builds_displ_Ux_32633.tif" #horizontal component.


def building_velocity(exposed_buildings, Uz_velocity_raster_filepath, Ux_velocity_raster_filepath, hazard_type='', object_id='OBJECTID', exposure='Exposure', mapfunc=mapclassify.MaximumBreaks):
    # Check for hazard_type in list of ground motion.
    if hazard_type not in ['landslide','subsidence','sinkhole']:
        raise Exception("Sorry, Hazard Type Not Allowed Yet!")
    # Slice and Reset index of gdf
    exp_builds = exposed_buildings.copy()
    exp_builds = exp_builds[[object_id,exposure,'geometry']]
    exp_builds = exp_builds[exposed_buildings[exposure] == 1]
    exp_builds = exp_builds.reset_index(drop=True)
    
    # Perform zonal statistics and Map VEL based on values distribution
    pct = [50, 82, 95, 97.5, 100]
    # LANDSLIDE
    if hazard_type == 'landslide':
        print('Analysing VEL for Landslide Hazard')
        # open raster as array (Uz + Ux)
        Uz_array, kwds_Uz, transform_Uz = open_raster(Uz_velocity_raster_filepath)
        Ux_array, kwds_Ux, transform_Ux = open_raster(Ux_velocity_raster_filepath)
        # zonal stats
        stats=['min','max']
        def abs_max(x):
            return np.abs(x).max()
        p_uz, p_ux = 'Uz', 'Ux'
        Uz_zs = zonal_stats(exp_builds, Uz_array, affine=transform_Uz, stats=stats,add_stats={'abs_max':abs_max}, prefix=p_uz)
        Ux_zs = zonal_stats(exp_builds, Ux_array, affine=transform_Ux, stats=stats,add_stats={'abs_max':abs_max}, prefix=p_ux)
        Uz_df = pd.DataFrame(Uz_zs).dropna().astype(float)
        Ux_df = pd.DataFrame(Ux_zs).dropna().astype(float)
        # Identify Maximum between Uz and Ux abs_max VEL
        maximum = np.maximum(Uz_df['Uzabs_max'], Ux_df['Uxabs_max'])
        max_vel = pd.DataFrame(maximum, columns=['max_vel'])
        # Map Classify to categorize VEL based on values distribution
        clf = mapfunc.make(pct=pct)
        # clf = mapclassify.Quantiles.make(k=5)
        max_vel['max_vel_class'] = max_vel[['max_vel']].apply(clf) +1
        # Merge with Geodataframe
        vel_gdf = exp_builds.merge(max_vel, left_index=True, right_index=True)
    
    # SUBSIDENCE
    elif hazard_type == 'subsidence':
        print('Analysing VEL for Subsidence Hazard')
        # Open Raster as Array (Uz)
        Uz_array, kwds_Uz, transform_Uz = open_raster(Uz_velocity_raster_filepath)
        stats = ['range']
        p_uz = 'Uz'
        Uz_zs = zonal_stats(exp_builds, Uz_array, affine=transform_Uz,stats=stats, prefix=p_uz)
        Uz_df = pd.DataFrame(Uz_zs).dropna().astype(float)
        # Map Classify to categorize VEL based on values distribution
        clf = mapfunc.make(pct=pct)
        Uz_df['Uzrange_class'] = Uz_df[['Uzrange']].apply(clf) +1
        # Merge with Geodataframe
        vel_gdf = exp_builds.merge(Uz_df, left_index=True, right_index=True)
    
    # SINKHOLE
    elif hazard_type == 'sinkhole':
        print('Analysing VEL for Sinkhole Hazard')
        # Open Raster as Array (Uz)
        Uz_array, kwds_Uz, transform_Uz = open_raster(Uz_velocity_raster_filepath)
        stats = ['min']
        p_uz = 'Uz'
        Uz_zs = zonal_stats(exp_builds, Uz_array, affine=transform_Uz,stats=stats, prefix=p_uz)
        Uz_df = pd.DataFrame(Uz_zs).dropna().astype(float)
        # Reverse Sign
        def Convert(lst):
            return [-i for i in lst]
        Uz_df['Uzmin_reverse'] = Convert(Uz_df['Uzmin'])
        # Merge with Geodataframe
        vel_gdf = exp_builds.merge(Uz_df, left_index=True, right_index=True)
        vel_gdf.drop(vel_gdf[vel_gdf['Uzmin']>0].index, inplace=True)
        # Map Classify to categorize VEL based on values distribution
        clf = mapfunc.make(pct=pct)
        vel_gdf['Uzmin_class'] = vel_gdf[['Uzmin_reverse']].apply(clf) +1
        # drop Uzmin_reverse
        vel_gdf.drop(columns=['Uzmin_reverse'], inplace=True)

    else: print('Entered Wrong Type of Hazard')
    return vel_gdf

# CALLING FUNCTION !!
vel_ls = building_velocity(exp_builds_landslide, filepath_Uz, filepath_Ux, hazard_type='landslide', mapfunc=mapclassify.Percentiles)
vel_subs = building_velocity(exp_builds_subsidence, filepath_Uz, filepath_Ux, hazard_type='subsidence', mapfunc=mapclassify.Percentiles)
vel_shsi = building_velocity(exp_builds_sinkhole, filepath_Uz, filepath_Ux, hazard_type='sinkhole', mapfunc=mapclassify.Percentiles)

# SAVE DATA
filepath = "PS-INSAR/Building_VEL"
vel_ls.to_file(os.path.join(filepath, 'build_vel_LS.gpkg'), driver='GPKG')
vel_subs.to_file(os.path.join(filepath, 'build_vel_SUBS.gpkg'), driver='GPKG')
vel_shsi.to_file(os.path.join(filepath, 'build_vel_SHSI.gpkg'), driver='GPKG')


#%% POTENTIAL DAMAGE SCORE
print('Computing Damage Scores...')
# ---------- VULNERABILITY ---------- #
# 1- Impact Ratio
def impact_ratio(gdf, raster_path, min_exposure_value=2, buffer=False, nodatavalue=-9999, pixel_size=5):
    """This function takes as input a geodataframe of polygons and a raster layer
    to compute ratio between exposed area and total area of each polygons"""
    # Open Raster as array   
    # compute categorical Zonal Statistics
    raster_array, kwds, transform = open_raster(raster_path)
    # replace nodata value with 0
    raster_array[raster_array == nodatavalue] = 0
    if buffer== True:
        # buffer analysis
        geom = gdf.geometry
        gdf['geometry'] = gdf.geometry.buffer(20, cap_style=3, join_style=2)
        # compute area of buildings
        gdf['total_area'] = gdf.area
        # zonal statistics (categorical)
        impact_zs = zonal_stats(gdf, raster_array, categorical=True, affine=transform)
        gdf.geometry = geom
    else:
        # compute area of buildings
        gdf['total_area'] = gdf.area
        impact_zs = zonal_stats(gdf, raster_array, categorical=True, affine=transform)
    # Compute Weighted Average of exposed number of pixels
    for i, dictionary in enumerate(impact_zs):
        total_classes = 0
        weighted_sum = 0
        for key, value in dictionary.items():
            if min_exposure_value <= key <= 5:  # Assuming that keys are numbers between 2 and 5
                total_classes += key
                weighted_sum += key * value
        if total_classes != 0:
            weighted_average = (weighted_sum / total_classes)
            gdf.loc[i, 'exposed_area'] = weighted_average*(pixel_size**2) # from pixel to area
        else:
            gdf.loc[i, 'exposed_area'] = 0
    # compute impact_ratio
    gdf['impact_ratio'] = gdf['exposed_area']/gdf['total_area']
    gdf['impact_ratio'] = np.where(gdf['impact_ratio']>1, 1, gdf['impact_ratio'])
    return gdf

# CALLING IMPACT_RATIO FUNC
builds = builds_w_all.loc[:, ['OBJECTID','edifc_uso_macro','VAL_Descr_Tipologia', 'c_parcel_id','geometry']]
impact_ls = impact_ratio(builds, filepath_ls, min_exposure_value=2, buffer=True)
impact_subs = impact_ratio(builds, filepath_subs, min_exposure_value=2, buffer=False)
impact_shsi = impact_ratio(builds, filepath_shsi, min_exposure_value=2, buffer=False)
    # MAP CLASSIFY
clf = mapclassify.UserDefined.make(bins=[0.001, 0.25, 0.50, 0.75, 1])
impact_ls['impact_class'] = impact_ls[['impact_ratio']].apply(clf)
impact_subs['impact_class'] = impact_subs[['impact_ratio']].apply(clf)
impact_shsi['impact_class'] = impact_shsi[['impact_ratio']].apply(clf)

# 2- Structural Resistance
cs_cols = ['id', 'sez2011', 'e1', 'e2', 'e3', 'e4','e5','e6','e7','e8','e9',
           'e10','e11','e12','e13','e14','e15','e16','e17','e18','e19','e20',
           'geometry']
# Census Tracts Data Cleaning
cs_clean = cs.loc[:, cs_cols]
cs_clean.replace(['null', None], 0, inplace=True)
cs_clean['e6'] = cs_clean['e6'].astype('float64')
cs_clean.fillna(0, inplace=True)

def building_resistance(df, sty_cols, sty_values, smn_cols, smn_values, sht_cols, sht_values):
    """This function takes census parcels as input and computes resistance 
    factors of buildings based on structural type, maintanance state and
    number of floors."""
    df_copy = df.copy()
    # fill nan
    df_copy.fillna(0, inplace=True)
    # Compute Resistance STY (Structural Type).
    df_copy['sty_res'] = ((df_copy[sty_cols[0]]*sty_values[0] +
                                         df_copy[sty_cols[1]]*sty_values[1] +
                                         df_copy[sty_cols[2]]*sty_values[2])
                                         / (df_copy[sty_cols[0]] + df_copy[sty_cols[1]] + df_copy[sty_cols[2]]
                                            )
                                         )
    df_copy['sty_no_res'] = 0.1
    
    # Compute Resistance SMN (Maintenance Type)
        # merge Age columns e8 - e16
    vp = df_copy[smn_cols[0]] + df_copy[smn_cols[1]]
    m = df_copy[smn_cols[2]] + df_copy[smn_cols[3]]
    g = df_copy[smn_cols[4]] + df_copy[smn_cols[5]]
    vg = df_copy[smn_cols[6]] + df_copy[smn_cols[7]] + df_copy[smn_cols[8]]
    
    df_copy['smn_res'] = ((vp*smn_values[0] +
                           m*smn_values[1] +
                           g*smn_values[2] +
                           vg*smn_values[3])
                          / (vp + m + g + vg)
                          )
    df_copy['smn_no_res'] = 0.1
    
    # Compute Resistance SHT (Number of floors)
    df_copy['sht_res'] = ((df_copy[sht_cols[0]]*sht_values[0] +
                                          df_copy[sht_cols[1]]*sht_values[1] +
                                          df_copy[sht_cols[2]]*sht_values[2] +
                                          df_copy[sht_cols[3]]*sht_values[3])
                                          / (df_copy[sht_cols[0]] + df_copy[sht_cols[1]] + df_copy[sht_cols[2]] + df_copy[sht_cols[3]]
                                            )
                                          )
    df_copy['sht_no_res'] = 0.1
    
    # Compute RESISTANCE FACTOR! (Li et al 2010)
    R_res = df_copy['sty_res']*df_copy['smn_res']*df_copy['sht_res']
    R_no_res = df_copy['sty_no_res']*df_copy['smn_no_res']*df_copy['sht_no_res']
    df_copy['R_res'] = R_res.apply(lambda x: math.pow(x, 1/3))
    df_copy['R_no_res'] = R_no_res.apply(lambda x: math.pow(x, 1/3))
    
    df_copy.fillna(0.1, inplace=True)
    return df_copy[['id','sty_res','sty_no_res','smn_res','smn_no_res','sht_res','sht_no_res','R_res','R_no_res','geometry']]

# Set cols and values
sty_cols = ['e5','e6','e7']
sty_values = [0.8, 1.5, 0.2]
smn_cols = ['e8','e9','e10','e11','e12','e13','e14','e15','e16']
smn_values = [0.1, 0.6, 1.2, 1.5]
sht_cols = ['e17','e18','e19','e20']
sht_values = [0.1, 0.4, 0.9, 1.5]
# Perform calculations
cs_resistance = building_resistance(cs_clean, sty_cols, sty_values, smn_cols, smn_values, sht_cols, sht_values)

# Assign values to buildings by census parcels and category of use (res/no_res)
    # merge builds with census parcels attributes
cols_to_drop = ['id','sty_res','sty_no_res','smn_res','smn_no_res','sht_res','sht_no_res', 'geometry_y']
builds_w_R = pd.merge(builds, cs_resistance, left_on='c_parcel_id', right_on='id').drop(columns=cols_to_drop).rename(columns={'geometry_x':'geometry'})
    # assign value of R for res and no_res buildings
builds_w_R['R'] = builds_w_R.apply(lambda row: row['R_res'] if row['VAL_Descr_Tipologia'] == 'Abitazioni civili' else row['R_no_res'], axis=1)
builds_w_R.drop(columns=['R_res','R_no_res'], inplace=True)

# MAP CLASSIFY R VALUES
clf_R = mapclassify.UserDefined.make(bins=[0,0.3, 0.5, 0.7, 1.0, 2.0])
builds_w_R['R_class'] = builds_w_R[['R']].apply(clf_R) - 1

# CONTINGENCY MATRIX AND ASSIGN VULNERABILITY VALUE
    # Merge IMPACT and RESISTANCE VALUES
        # LANDSLIDE
builds_V_ls = pd.merge(impact_ls, builds_w_R[['OBJECTID','R','R_class']], on='OBJECTID')
        # SUBSIDENCE
builds_V_subs = pd.merge(impact_subs, builds_w_R[['OBJECTID','R','R_class']], on='OBJECTID')
        # SINKHOLE
builds_V_shsi = pd.merge(impact_shsi, builds_w_R[['OBJECTID','R','R_class']], on='OBJECTID')
    # Read Contingency Matrix
cm = "ASSETS/V_contingency_matrix.xlsx"
cm_df = pd.read_excel(cm)
contingency_matrix = cm_df.set_index(['impact_class', 'R_class'])['vulnerability'].to_dict()
    # Apply Contingency Matrix
builds_V_ls['vulnerability'] = builds_V_ls.apply(lambda row: contingency_matrix.get((row['impact_class'], row['R_class'])), axis=1)

builds_V_subs['vulnerability'] = builds_V_subs.apply(lambda row: contingency_matrix.get((row['impact_class'], row['R_class'])), axis=1)

builds_V_shsi['vulnerability'] = builds_V_shsi.apply(lambda row: contingency_matrix.get((row['impact_class'], row['R_class'])), axis=1)

# ----- ASSETS AVERAGE MARKET VALUE -----#
builds_w_all['Average_Value'] = (builds_w_all['VAL_Compr_min'] + builds_w_all['VAL_Compr_max']) / 2
builds_value = builds_w_all[['OBJECTID', 'CODZONA','Average_Value']].copy()

# ----- POTENTIAL DAMAGE -----#
  # 1- LANDSLIDE
cols_to_keep = ['OBJECTID','CODZONA','edifc_uso_macro','VAL_Descr_Tipologia','c_parcel_id',
                'vulnerability','Average_Value', 'exposed_area', 'impact_ratio','impact_class',
                'R', 'R_class',
                'geometry']
builds_damage_ls = pd.merge(builds_V_ls, builds_value, on='OBJECTID')[cols_to_keep]
builds_damage_ls['damage'] = (builds_damage_ls['vulnerability'] * builds_damage_ls['Average_Value'])
  # 2- SUBSIDENCE
builds_damage_subs = pd.merge(builds_V_subs, builds_value, on='OBJECTID')[cols_to_keep]
builds_damage_subs['damage'] = (builds_damage_subs['vulnerability'] * builds_damage_subs['Average_Value'])
  # 3- SINKHOLE
builds_damage_shsi = pd.merge(builds_V_shsi, builds_value, on='OBJECTID')[cols_to_keep]
builds_damage_shsi['damage'] = (builds_damage_shsi['vulnerability'] * builds_damage_shsi['Average_Value'])
# Save Results
cs_resistance.to_file('DAMAGE/cs_resistance.gpkg', driver='GPKG')
builds_damage_ls.to_file('DAMAGE/damage_landslide.gpkg', driver='GPKG')
builds_damage_subs.to_file('DAMAGE/damage_subsidence.gpkg', driver='GPKG')
builds_damage_shsi.to_file('DAMAGE/damage_sinkhole.gpkg', driver='GPKG')


#%% Modelling Single-Hazard Risk
print('Computing Hazard-Specific Risk Scores...')

def compute_specific_risk(Hazard, Activity, Damage, ID_col='OBJECTID', h_type=''):
    if h_type == 'landslide':
        # Hazard Score
        H_score = Hazard[[ID_col, 'ls_max']].set_index(ID_col)
        H_score.rename(columns={'ls_max':'hazard_class'}, inplace=True)
        # Activity Score
        A_score = Activity[[ID_col, 'max_vel_class', 'geometry']].set_index(ID_col).rename(columns={'max_vel_class':'activity_class'})
    elif h_type == 'subsidence':
        H_score = Hazard[[ID_col, 'subs_max']].set_index(ID_col)
        H_score.rename(columns={'subs_max':'hazard_class'}, inplace=True)
        # Activity Score
        A_score = Activity[[ID_col, 'Uzrange_class', 'geometry']].set_index(ID_col).rename(columns={'Uzrange_class':'activity_class'})
    elif h_type == 'sinkhole':
        H_score = Hazard[[ID_col, 'shsi_max']].set_index(ID_col)
        H_score.rename(columns={'shsi_max':'hazard_class'}, inplace=True)
        # Activity Score
        A_score = Activity[[ID_col, 'Uzmin_class', 'geometry']].set_index(ID_col).rename(columns={'Uzmin_class':'activity_class'})
    else: raise Exception("Sorry, Hazard Type Not Allowed Yet!")
    
    # Damage Score
    D_score = Damage.set_index(ID_col).drop(columns='geometry')
    
    # Compute Hazard-Specific Risk Class
    Risk = A_score.join([H_score, D_score], how='left')
        # Map Classify Damage for Exposed buildings only
    clf = mapclassify.Percentiles.make(pct=[20, 40, 60, 80, 100])
    Risk['damage_class'] = Risk[['damage']].apply(clf) +1
    Risk['Risk_{0}'.format(h_type)] = Risk['hazard_class']*Risk['activity_class']*Risk['damage_class']
    
    return Risk

# LANDSLIDE RISK
Hazard = exp_builds_landslide
Activity = vel_ls
Damage = builds_damage_ls
RISK_LS = compute_specific_risk(Hazard, Activity, Damage, 'OBJECTID', 'landslide')

# SUBSIDENCE RISK
Hazard = exp_builds_subsidence
Activity = vel_subs
Damage = builds_damage_subs
RISK_SUBS = compute_specific_risk(Hazard, Activity, Damage, 'OBJECTID', 'subsidence')

# SINKHOLE RISK
Hazard = exp_builds_sinkhole
Activity = vel_shsi
Damage = builds_damage_shsi
RISK_SHSI = compute_specific_risk(Hazard, Activity, Damage, 'OBJECTID', 'sinkhole')

# SAVE RESULTS
RISK_LS.to_file('RESULTS/Risk_LS_new.gpkg', driver='GPKG')
RISK_SUBS.to_file('RESULTS/Risk_SUBS_new.gpkg', driver='GPKG')
RISK_SHSI.to_file('RESULTS/Risk_SHSI_new.gpkg', driver='GPKG')

# Modelling Multi-Hazard Risk
print('Computing Multi-Hazard Risk Scores...')

# MERGE SPECIFICK RISKS INTO SINGLE GDF.
h1, h2, h3 = 'landslide','subsidence','sinkhole'
cols_to_keep1 = ['OBJECTID','activity_class','hazard_class','vulnerability','Average_Value','damage','damage_class',f'Risk_{h1}']
cols_to_keep2 = ['OBJECTID','activity_class','hazard_class','vulnerability','Average_Value','damage','damage_class',f'Risk_{h2}']
cols_to_keep3 = ['OBJECTID','activity_class','hazard_class','vulnerability','Average_Value','damage','damage_class',f'Risk_{h3}']
RISK_LS   = RISK_LS.reset_index()[cols_to_keep1].add_suffix('_LS')
RISK_SUBS = RISK_SUBS.reset_index()[cols_to_keep2].add_suffix('_SUBS')
RISK_SHSI = RISK_SHSI.reset_index()[cols_to_keep3].add_suffix('_SHSI')
del h1, h2, h3, cols_to_keep1,cols_to_keep2,cols_to_keep3

RISK = builds_w_all.merge(RISK_LS, left_on='OBJECTID',right_on='OBJECTID_LS',how='left', validate='1:1')
RISK = RISK.merge(RISK_SUBS, left_on='OBJECTID',right_on='OBJECTID_SUBS',how='left', validate='1:1')
RISK = RISK.merge(RISK_SHSI, left_on='OBJECTID',right_on='OBJECTID_SHSI',how='left', validate='1:1')
RISK = RISK.drop(columns=['OBJECTID_LS','OBJECTID_SUBS','OBJECTID_SHSI'])

#### COMPUTE MULTI-HAZARD RISK PRIORITY SCORE ####
    # SHOCK / STRESS FACTOR F1 (shock=1, stress=0.5)
RISK['MH_R'] = (RISK['Risk_landslide_LS'].fillna(0)*1 + 
              RISK['Risk_subsidence_SUBS'].fillna(0)*0.5 + 
              RISK['Risk_sinkhole_SHSI'].fillna(0)*1)

# DOMINANT HAZARD IDENTIFICATION AND QUANTIFICATION
# derive maximum value across specific risk types
max_values = RISK[['Risk_landslide_LS','Risk_subsidence_SUBS','Risk_sinkhole_SHSI']].max(axis=1)
# assign string corresponding to max value
conditions = [max_values == RISK['Risk_landslide_LS'],
              max_values == RISK['Risk_subsidence_SUBS'],
              max_values == RISK['Risk_sinkhole_SHSI'],
              pd.isnull(max_values)
              ]
choices = ['Landslide', 'Subsidence', 'Sinkhole', None]
RISK['max_r_type'] = np.select(conditions, choices)


# map classify specific Risk and MH_R values
clf = mapclassify.Percentiles.make(pct=[50,82,95,97.5,100])
mh_r = RISK[RISK['MH_R']>0].copy()
mh_r['MH_R_class'] = mh_r[['MH_R']].apply(clf)+1

RISK = RISK.merge(mh_r[['MH_R_class', 'OBJECTID']], on='OBJECTID', how='left')

# SAVE RESULTS !!
RISK.to_file('RESULTS/MH_RISK_new.gpkg', driver='GPKG')

