@author: giand
"""

# import libraries
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
AOI
OMI
cs
builds_w_all
