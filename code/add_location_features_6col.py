# -*- coding: utf-8 -*-
"""
6개 컬럼 Boolean 방식 (성능 비교용)
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def check_bus_service_zones_6col(df, bus_df):
    """6개 컬럼 방식: 버스정거장 Boolean 컬럼"""
    df = df.copy()
    bus_df = bus_df.copy()
    bus_df = bus_df.dropna(subset=['X좌표', 'Y좌표'])
    bus_df['X좌표'] = pd.to_numeric(bus_df['X좌표'], errors='coerce')
    bus_df['Y좌표'] = pd.to_numeric(bus_df['Y좌표'], errors='coerce')
    bus_df = bus_df.dropna(subset=['X좌표', 'Y좌표'])
    
    print(f"버스정거장 데이터: {len(bus_df)}개")
    
    bus_coords_rad = np.column_stack([
        np.radians(bus_df['X좌표'].values),
        np.radians(bus_df['Y좌표'].values)
    ])
    
    R = 6371.0
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine')
    nn_model.fit(bus_coords_rad)
    
    df_with_coords = df[df['좌표X'].notna() & df['좌표Y'].notna()].copy()
    df_without_coords = df[df['좌표X'].isna() | df['좌표Y'].isna()].copy()
    
    bus_features_dict = {}
    
    if len(df_with_coords) > 0:
        apt_coords_rad = np.column_stack([
            np.radians(df_with_coords['좌표X'].values),
            np.radians(df_with_coords['좌표Y'].values)
        ])
        
        distances_rad, indices = nn_model.kneighbors(apt_coords_rad)
        distances_km = distances_rad.flatten() * R
        
        for orig_idx, dist_km in zip(df_with_coords.index, distances_km):
            dist_m = dist_km * 1000
            
            bus_features_dict[orig_idx] = {
                '버스_핵심서비스권': dist_m <= 300,
                '버스_일반영향권': (dist_m > 300) & (dist_m <= 500)
            }
    
    for orig_idx in df_without_coords.index:
        bus_features_dict[orig_idx] = {
            '버스_핵심서비스권': False,
            '버스_일반영향권': False
        }
    
    bus_features_df = pd.DataFrame([bus_features_dict[idx] for idx in df.index], index=df.index)
    df = pd.concat([df, bus_features_df], axis=1)
    
    return df

def check_subway_zones_6col(df, subway_df):
    """6개 컬럼 방식: 지하철역 Boolean 컬럼"""
    df = df.copy()
    subway_df = subway_df.copy()
    subway_df = subway_df.dropna(subset=['경도', '위도'])
    subway_df['경도'] = pd.to_numeric(subway_df['경도'], errors='coerce')
    subway_df['위도'] = pd.to_numeric(subway_df['위도'], errors='coerce')
    subway_df = subway_df.dropna(subset=['경도', '위도'])
    
    print(f"지하철역 데이터: {len(subway_df)}개")
    
    subway_coords_rad = np.column_stack([
        np.radians(subway_df['경도'].values),
        np.radians(subway_df['위도'].values)
    ])
    
    R = 6371.0
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine')
    nn_model.fit(subway_coords_rad)
    
    df_with_coords = df[df['좌표X'].notna() & df['좌표Y'].notna()].copy()
    df_without_coords = df[df['좌표X'].isna() | df['좌표Y'].isna()].copy()
    
    subway_features_dict = {}
    
    if len(df_with_coords) > 0:
        apt_coords_rad = np.column_stack([
            np.radians(df_with_coords['좌표X'].values),
            np.radians(df_with_coords['좌표Y'].values)
        ])
        
        distances_rad, indices = nn_model.kneighbors(apt_coords_rad)
        distances_km = distances_rad.flatten() * R
        
        for orig_idx, dist_km in zip(df_with_coords.index, distances_km):
            dist_m = dist_km * 1000
            
            subway_features_dict[orig_idx] = {
                '지하철_1차역세권': dist_m <= 350,
                '지하철_2차역세권': (dist_m > 350) & (dist_m <= 500),
                '지하철_초역세권': dist_m <= 300,
                '지하철_역세권': (dist_m > 300) & (dist_m <= 500)
            }
    
    for orig_idx in df_without_coords.index:
        subway_features_dict[orig_idx] = {
            '지하철_1차역세권': False,
            '지하철_2차역세권': False,
            '지하철_초역세권': False,
            '지하철_역세권': False
        }
    
    subway_features_df = pd.DataFrame([subway_features_dict[idx] for idx in df.index], index=df.index)
    df = pd.concat([df, subway_features_df], axis=1)
    
    return df

def add_location_features_6col(df, bus_csv_path, subway_csv_path):
    """6개 컬럼 Boolean 방식"""
    print("=" * 60)
    print("역세권 여부 확인 시작 (6개 컬럼 Boolean 방식)")
    print("=" * 60)
    
    bus_df = pd.read_csv(bus_csv_path, encoding='utf-8')
    subway_df = pd.read_csv(subway_csv_path, encoding='utf-8')
    
    print(f"\n1. 버스정거장 서비스권역 여부 확인 중...")
    df = check_bus_service_zones_6col(df, bus_df)
    
    print(f"\n2. 지하철역 역세권 여부 확인 중...")
    df = check_subway_zones_6col(df, subway_df)
    
    print("\n" + "=" * 60)
    print("역세권 여부 확인 완료! (6개 컬럼 Boolean 방식)")
    print("=" * 60)
    
    return df
