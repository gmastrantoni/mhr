#%% IMPORT
import geopandas as gpd
# from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.ops import nearest_points
# from shapely.ops import nearest_points

from tobler.area_weighted import area_interpolate
# from tobler.dasymetric import masked_area_interpolate

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx

import rasterio
from rasterio.plot import show
from rasterstats import zonal_stats

import os
os.chdir(".../data")

### Reading files
# Boundary of the area of interest
AOI = gpd.read_file("AOI.gpkg", ignore_fields=['location', 'id', 'objectid', 'cod_pro',
                                               'cod_istat', 'pro_com', 'nome', 'shape_leng', 'shape_area'])

# Census Tracts
census = gpd.read_file("ASSETS/sez_censuarie_2011_Roma_32633.gpkg",
                       ignore_fields=['pro_com', 'cod_stagno','cod_fiume', 
                                      'cod_lago', 'cod_laguna', 'cod_val_p', 'cod_zona_c',
                                      'cod_is_amm', 'cod_is_lac', 'cod_is_mar', 'cod_area_s', 'cod_mont_d',
                                      'loc2011', 'cod_loc', 'tipo_loc', 'com_asc', 'cod_asc', 'ace',
                                      'shape_leng'
                                      ]
                       )

# OMI Zones
omi = gpd.read_file("ASSETS/OMI_Zone_Valori_32633.gpkg",
                    ignore_fields=['LINKZONA', 'CODCOM', 'ZONE_Comune_descrizione',
                           'ZONE_Zona_Descr', 'ZONE_Zona', 'ZONE_LinkZona', 'ZONE_Stato_prev', 'ZONE_Microzona',
                           'VAL_Zona', 'VAL_Fascia', 'VAL_Stato_prev', 'VAL_Sup_NL_compr', 
                           'VAL_Loc_min', 'VAL_Loc_max', 'VAL_Sup_NL_loc'
                           ]
                    )

# Buildings
BUILD = gpd.read_file("ASSETS/dbsn_edifici_AOI_32633.gpkg",
                      ignore_fields=['edifc_sot', 'classid', 'edifc_nome', 'edifc_stat', 
                                     'edifc_at', 'scril', 'meta_ist', 'edifc_mon', 
                                     'shape_Length', 'shape_Area']
                      ).set_index('OBJECTID')

#%% OVERLAY with AOI
mun_aoi = gpd.gpd.clip(mun, AOI)
census_aoi = gpd.clip(census, AOI)
omi_aoi = gpd.clip(omi, AOI)
# delete original gdf
del mun, census, omi

#%% FUNCTIONS

# Merge categories of "edifc_uso"
def merge_category(cat):
    macro_cats = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '95', '93']
    for macro in macro_cats:
        if cat.startswith(macro):
            return macro
        else:
            pass

def fill_missing_values(gdf, col, k=3):
    # creiamo una copia del geodataframe per evitare di modificare l'originale
    filled_gdf = gdf.copy()
    
    # selezioniamo i record con valore nullo nella colonna "VAL_Compr_min"
    null_rows = filled_gdf[filled_gdf[col].isnull()]
        
    # iteriamo sui record con valore nullo
    for idx, row in null_rows.iterrows():
        # selezioniamo i tre poligoni più vicini al punto di interesse
        point = Point(row["geometry"].centroid.x, row["geometry"].centroid.y)
        distances = filled_gdf.distance(point)
        nearest_poly_indices = distances.sort_values().index[:k]
        
        # calcoliamo la media dei valori "VAL_Compr_min" dei tre poligoni vicini
        nearest_vals = filled_gdf[~filled_gdf.index.isin([idx]) & filled_gdf.index.isin(nearest_poly_indices)][col]
        mean_val = nearest_vals.mean()
        
        # sostituiamo il valore nullo con la media dei valori dei poligoni vicini
        filled_gdf.loc[idx, col] = mean_val
        # check remaining NANs and fill them with mean value of the series.
        # filled_gdf.fillna()
    return filled_gdf

def fill_missing_values_by_type(gdf, col, col_type='VAL_Descr_Tipologia', k=3):
    # creo empty geodataframe
    result_gdf = gpd.GeoDataFrame(crs='EPSG:32633')
    # iterate on types
    types = gdf[col_type].unique()
    for t in types:
        filled_gdf_type = gdf.loc[gdf[col_type] == t]
        # selezioniamo i record con valore nullo nella colonna "VAL_Compr_min"
        null_rows = filled_gdf_type.loc[filled_gdf_type[col].isnull()]
        # iteriamo sui record con valore nullo
        for idx, row in null_rows.iterrows():
            # selezioniamo i tre poligoni più vicini al punto di interesse
            point = Point(row["geometry"].centroid.x, row["geometry"].centroid.y)
            distances = filled_gdf_type.distance(point)
            nearest_poly_indices = distances.sort_values().index[:k]
            
            # calcoliamo la media dei valori "VAL_Compr_min" dei tre poligoni vicini
            nearest_vals = filled_gdf_type[~filled_gdf_type.index.isin([idx]) & filled_gdf_type.index.isin(nearest_poly_indices)][col]
            mean_val = nearest_vals.mean()
            
            # sostituiamo il valore nullo con la media dei valori dei poligoni vicini
            filled_gdf_type.loc[idx, col] = mean_val
            # check remaining NANs and fill them with mean value of the series.
        mean_by_col_type = filled_gdf_type[col].mean()
        filled_gdf_type[col].fillna(mean_by_col_type, inplace=True)
    
        # append to results
        result_gdf = result_gdf.append(filled_gdf_type)
    return result_gdf

# Spatial joint between polygon with the largest area of intersection
def largest_intersection(gdf_left, gdf_right, mode):
    """
    Take two geodataframes, do a spatial join, and return the polygon 
    with the largest area of intersection
    """
    out_gdf = gpd.sjoin(gdf_left, gdf_right, how = "left", predicate = mode).dropna()
    out_gdf['intersection'] = [a.intersection(gdf_right[gdf_right.index == b].geometry.values[0]).area for a, b in zip(out_gdf.geometry.values, out_gdf.index_right)]
    out_gdf['index'] = out_gdf.index
    out_gdf = out_gdf.sort_values(by='intersection')
    out_gdf = out_gdf.drop_duplicates(subset = 'index', keep='last')
    out_gdf = out_gdf.sort_values(by='index')
    out_gdf = out_gdf.drop(columns=['index_right', 'intersection', 'index'])
    
    return out_gdf


def open_raster(raster_file_path):
    with rasterio.open(raster_file_path) as src:
        #get transform
        transform = src.meta['transform']
        #create 2darray from band 1
        array = src.read(1)
        print("Array shape: ", array.shape)
        
        print('src type: ', type(src))
        # get raster profile
        print('no data value: ', src.nodata)
        kwds = src.profile

        return array, kwds, transform

def save_raster(raster_array, name, kwds):
    with rasterio.open(os.path.join(name+".tif"), 'w', **kwds) as dst:
        dst.write(raster_array, indexes=1)

#%% Pre-processing
## SPATIAL FILL NAN --> Func fill_missing_values
most_fr = omi_aoi['VAL_Descr_Tipologia'].mode().iloc[0]
omi_aoi['VAL_Descr_Tipologia'].fillna(most_fr, inplace=True)
omi_20 = omi_aoi[omi_aoi['VAL_Descr_Tipologia'].isin(['Abitazioni civili', 'Ville e Villini'])]
omi_20_fill = fill_missing_values(omi_20, "VAL_Compr_min", k=3)
omi_20_fill_ok = fill_missing_values(omi_20_fill, "VAL_Compr_max", k=3)
omi_20_nan = omi_20_fill_ok[omi_20_fill_ok['VAL_Cod_Tip'].isna()]
# assign filled values to original omi_aoi
omi_aoi['VAL_Compr_min'].fillna(omi_20_nan['VAL_Compr_min'], inplace=True)
omi_aoi['VAL_Compr_max'].fillna(omi_20_nan['VAL_Compr_max'], inplace=True)
del most_fr, omi_20, omi_20_fill, omi_20_fill_ok, omi_20_nan
# save omi_aoi
omi_aoi.to_file("ASSETS/OMI_AOI.gpkg", driver='GPKG')


## Spatial Join BUILD-OMI 
# select unique omi polygons with codzona attribute
omi_codzona = omi_aoi[['CODZONA', 'geometry']].sort_index().drop_duplicates(subset=['CODZONA', 'geometry'])
# spatial join one-to-one BUILD with omi_codzona
builds_codzona = largest_intersection(BUILD, omi_codzona, 'intersects')
## Buildings Macro Categories
builds_codzona['edifc_uso_macro'] = builds_codzona['edifc_uso'].apply(merge_category)
builds_codzona = builds_codzona.reset_index()
del BUILD
# save builds_codzona
builds_codzona.to_file("ASSETS/builds_w_codzona.gpkg", driver='GPKG')


# MAP Cod_Tip and Edific_Uso
# Matches values of edifc_uso, VAL_Cod_Tip
edifc_uso_map = {'01':'Abitazioni civili',
                 '02':'Uffici',
                 # '03': 'Abitazioni di tipo economico',
                 '03': 'Uffici',
                 '04': 'Uffici',
                 '05': '-',
                 '06': 'Abitazioni di tipo economico',
                 '07': 'Negozi',
                 '08': 'Capannoni industriali',
                 '09': 'Capannoni tipici',
                 '10': 'Abitazioni di tipo economico', # o abitazioni civili
                 '11': 'Abitazioni di tipo economico',
                 '12': 'Abitazioni di tipo economico', # o abitazioni civili
                 '95':'-',
                 '93':'-'}
# map
builds_codzona['VAL_Descr_Tipologia'] = builds_codzona['edifc_uso_macro'].map(edifc_uso_map)
# drop '-'
builds_codzona = builds_codzona.drop(builds_codzona[builds_codzona['VAL_Descr_Tipologia'] == '-'].index)

# MERGE [VAL_Compr_min, VAL_Compr_max] on [CODZONA, VAL_Cod_Tip]
builds_w_val = builds_codzona.merge(omi_aoi, on=['CODZONA','VAL_Descr_Tipologia'], how='left',
                                    ).drop(columns=['geometry_y']
                                           # ).set_index('OBJECTID'
                                           ).rename(columns={'geometry_x': 'geometry'}
                                                    ).set_geometry('geometry')
                                                    # droppa colonne inutili
# sort values and drop_duplicates keeping last
builds_w_val = builds_w_val.sort_values(by=['OBJECTID', 'VAL_Compr_min']
                                        ).drop_duplicates(subset=['OBJECTID'], keep='last')

# Fill NAN with KNN
#check NAN
print(builds_w_val.isna().sum())

builds_w_val_filled = fill_missing_values_by_type(builds_w_val, 'VAL_Compr_min', col_type='VAL_Descr_Tipologia', k=10)
builds_w_val_filled2 = fill_missing_values_by_type(builds_w_val_filled, 'VAL_Compr_max', col_type='VAL_Descr_Tipologia', k=10)
del builds_w_val_filled
#drop useless cols
cols_to_drop = ['Name', 'ZONE_Fascia', 'ZONE_Cod_tip_prev', 'ZONE_Descr_tip_prev', 'VAL_Cod_Tip', 'VAL_Stato']
builds_w_val_filled_ok = builds_w_val_filled2.drop(cols_to_drop, axis=1)
# drop nan
builds_w_val_filled_ok.dropna(inplace=True)
# save builds_w_val
builds_w_val_filled_ok.to_file("ASSETS/builds_w_val_filled_by_type_new.gpkg", driver='GPKG')
# Filter on condition
condition = ((builds_w_val_filled_ok['edifc_uso_macro'] == '10') & builds_w_val_filled_ok['edifc_ty'].isin(['11', '12', '15', '16', '95']))
builds_w_omi = builds_w_val_filled_ok.drop(builds_w_val_filled_ok[condition].index)
#save gdf
builds_w_omi.to_file("ASSETS/builds_w_omi_all.gpkg", driver='GPKG')

#%% SJOIN BUILDS w CENSUS
# spatial join one-to-one between buildings and census.
cols_of_interest = ['id', 'geometry']
cens_id = census_aoi[cols_of_interest].rename(columns={'id':'c_parcel_id'})
builds_w_all = largest_intersection(builds_w_omi, cens_id, 'intersects')
# drop rows with VAL_Descr_Tipologia == '-'.
builds_w_all = builds_w_all.drop(builds_w_all[builds_w_all['VAL_Descr_Tipologia'] == '-'].index)
#save gdf
builds_w_all.to_file("ASSETS/builds_w_all.gpkg", driver='GPKG')
