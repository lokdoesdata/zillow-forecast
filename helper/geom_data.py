"""Geographical Data

Help functions used to process the ESRI and Census files used for Zillow
forecasting.

Author: Lok Ngan (lokdoesdata)
"""

# Package needed
import pandas as pd
import geopandas as gpd

# Vanilla Python
import functools
import multiprocessing
import json
from urllib.request import (
    urlopen,
    urlretrieve
)
import zipfile
from pathlib import Path

STATES = [
    'AL', 'AK', 'AR', 'AS', 'AZ',
    'CA', 'CO', 'CT',
    'DC', 'DE',
    'FL',
    'GA', 'GU',
    'HI',
    'IA', 'ID', 'IL', 'IN',
    'KS', 'KY',
    'LA',
    'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT',
    'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
    'OH', 'OK', 'OR',
    'PA', 'PR',
    'RI',
    'SC', 'SD',
    'TN', 'TX',
    'UT',
    'VA', 'VI', 'VT',
    'WA', 'WI', 'WV', 'WY'
]  # This also include U.S. Territories

ESRI_LIMIT = 2000

DATA_PATH = Path(__file__).parents[1].joinpath('data').absolute()
AREA_PATH = DATA_PATH.joinpath('usps_zip_code_area')
POINT_PATH = DATA_PATH.joinpath('usps_zip_code_point')
MSA_PATH = DATA_PATH.joinpath('msa')

AREA_PATH.mkdir(parents=True, exist_ok=True)
POINT_PATH.mkdir(parents=True, exist_ok=True)
MSA_PATH.mkdir(parents=True, exist_ok=True)

ARC_URL = r'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/'

AREA_URL = ARC_URL + r'USA_ZIP_Code_Areas_anaylsis/FeatureServer/0/query?'
AREA_URL = AREA_URL + r'f=geojson&outFields=*&where=STATE='

POINT_URL = ARC_URL + r'USA_ZIP_Code_Points_analysis/FeatureServer/0/query?'
POINT_URL = POINT_URL + r'f=geojson&outFields=*&where=STATE='

MSA_URL = r'https://www2.census.gov/geo/tiger/GENZ2010/'
MSA_URL = MSA_URL + r'gz_2010_us_310_m1_500k.zip'

del ARC_URL


def get_msa_gdf():
    out_file = MSA_URL.joinpath('gz_2010_us_310_m1_500k.zip')

    if not out_file.is_file():
        urlretrieve(MSA_URL, out_file)

    return(gpd.read_file(out_file))


def _combine_and_compress_geojson(geojson, output_file):

    out_features = []

    for j in geojson:
        out_features.extend(j['features'])

    out_dictionary = dict(type='FeatureCollection', features=out_features)
    out_geojson = json.dumps(out_dictionary).encode('utf-8')

    with zipfile.ZipFile(
            output_file, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        with z.open(output_file.name[:-4], 'w') as out:
            out.write(out_geojson)


def get_zip_code_gdf_by_state(state, geom_type='area'):

    if geom_type == 'area':
        url = f"{AREA_URL}'{state}'&resultOffset="
        out_file = AREA_PATH.joinpath(
            f'{state}_usps_zip_code_area.geojson.zip')
    elif geom_type == 'point':
        url = f"{POINT_URL}'{state}'&resultOffset="
        out_file = POINT_PATH.joinpath(
            f'{state}_usps_zip_code_point.geojson.zip')
    else:
        msg = 'geom_type can only be area or point'
        raise Exception(msg)

    # if file exists, read and return file as GeoDataFrame
    if out_file.is_file():
        return(gpd.read_file(out_file))

    offset = 0
    temp_data = []

    while True:
        temp_url = url + str(offset*ESRI_LIMIT)

        with urlopen(temp_url) as resp:
            temp_json = json.load(resp)

        temp_data.append(temp_json)

        if temp_json.get('properties') is None:
            break
        if temp_json.get('properties').get('exceededTransferLimit') is None:
            break

        offset += 1

    _combine_and_compress_geojson(temp_data, out_file)

    if out_file.is_file():
        return(gpd.read_file(out_file))


def get_zip_code_gdf():

    # Use pool to generate gdf for zip code points and areas quickly
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    gdf_point = pool.map(
        functools.partial(
            get_zip_code_gdf_by_state, geom_type='point'), STATES)

    gdf_area = pool.map(
        functools.partial(
            get_zip_code_gdf_by_state, geom_type='area'), STATES)

    pool.close()

    # Concat the pieces from zip code
    gdf_point = pd.concat(gdf_point)[[
        'ZIP_CODE', 'PO_NAME', 'STATE', 'POPULATION', 'SQMI', 'geometry']]

    gdf_area = pd.concat(gdf_area)[[
        'ZIP_CODE', 'PO_NAME', 'STATE', 'POPULATION', 'SQMI', 'geometry']]

    gdf_msa = get_msa_gdf()[['NAME', 'geometry']]
    gdf_msa = gdf_msa.to_crs(4326)

    # intersect this with the MSA gdf
    gdf_area_intersect = gpd.overlay(
        gdf_msa,
        gdf_area[['ZIP_CODE', 'SQMI', 'geometry']],
        how='intersection'
    )

    # calculate the intersecting area
    gdf_area_intersect = gdf_area_intersect.to_crs({'proj': 'cea'})
    gdf_area_intersect['intersecting_area'] = (
        gdf_area_intersect.area/2589988.1103)

    # Find the max zip code/MSA pair
    gdf_area_intersect = gdf_area_intersect[
        gdf_area_intersect.groupby(
            ['ZIP_CODE'])['intersecting_area'].transform(
                max) == gdf_area_intersect['intersecting_area']
    ]

    # Find the percent area intersect
    gdf_area_intersect['intersecting_percentage'] = round(
        gdf_area_intersect['intersecting_area'] /
        gdf_area_intersect['SQMI'], 3)

    # If intersecting area is more than 50%, then consider zip code within
    # MSA
    gdf_area_intersect = gdf_area_intersect[(
        gdf_area_intersect['intersecting_percentage'] > 0.5)]
    gdf_area_intersect = gdf_area_intersect[['ZIP_CODE', 'NAME']]

    gdf_area = gdf_area.merge(gdf_area_intersect, how='left', on='ZIP_CODE')

    # Find point zip code within a MSA
    gdf_point = gdf_point[~gdf_point['ZIP_CODE'].isin(gdf_area['ZIP_CODE'])]
    gdf_point = gpd.sjoin(
        gdf_point, gdf_msa,
        how='left', op='within').drop('index_right', axis=1)

    gdf = pd.concat([gdf_area, gdf_point])

    return(gdf)
