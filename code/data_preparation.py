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

