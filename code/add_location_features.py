# -*- coding: utf-8 -*-
"""
아파트 데이터에 버스정거장 및 지하철역 역세권 여부 추가
실거래가에 영향을 미칠 수 있는 역세권 범위 내 포함 여부만 확인
NearestNeighbors 모델을 사용하여 성능 최적화
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Haversine 거리 메트릭은 NearestNeighbors에서 직접 사용
# 별도 함수 불필요 (NearestNeighbors의 metric='haversine' 사용)


def check_bus_service_zones(df, bus_df):
    """
    각 아파트에서 버스정거장 서비스권역 확인 (Boolean 값으로 표시)
    - 핵심서비스권: 300m 이내
    - 일반영향권: 300m~500m 이내
    
    Parameters:
    -----------
    df : DataFrame
        아파트 데이터 (좌표X, 좌표Y 컬럼 필요)
    bus_df : DataFrame
        버스정거장 데이터 (X좌표, Y좌표 컬럼 필요)
    
    Returns:
    --------
    DataFrame : 원본 데이터에 버스 서비스권역이 추가된 데이터
    """
    df = df.copy()
    
    # 버스정거장 데이터 전처리
    bus_df = bus_df.copy()
    bus_df = bus_df.dropna(subset=['X좌표', 'Y좌표'])
    bus_df['X좌표'] = pd.to_numeric(bus_df['X좌표'], errors='coerce')
    bus_df['Y좌표'] = pd.to_numeric(bus_df['Y좌표'], errors='coerce')
    bus_df = bus_df.dropna(subset=['X좌표', 'Y좌표'])
    
    print(f"버스정거장 데이터: {len(bus_df)}개")
    print(f"서비스권역 범위:")
    print(f"  핵심서비스권: 300m 이내")
    print(f"  일반영향권: 300m~500m 이내")
    
    # 좌표를 라디안으로 변환 (Haversine 거리를 위해)
    bus_coords_rad = np.column_stack([
        np.radians(bus_df['X좌표'].values),  # 경도
        np.radians(bus_df['Y좌표'].values)   # 위도
    ])
    
    # NearestNeighbors 모델 생성 및 학습
    print("NearestNeighbors 모델 학습 중...")
    R = 6371.0  # 지구 반지름 (km)
    
    # 가장 가까운 버스정거장의 거리를 확인하기 위해 모델 생성
    nn_model = NearestNeighbors(
        n_neighbors=1,  # 가장 가까운 1개 확인
        algorithm='ball_tree',
        metric='haversine'
    )
    nn_model.fit(bus_coords_rad)
    
    # 아파트 좌표 전처리
    df_with_coords = df[df['좌표X'].notna() & df['좌표Y'].notna()].copy()
    df_without_coords = df[df['좌표X'].isna() | df['좌표Y'].isna()].copy()
    
    # 결과 저장용 딕셔너리
    bus_features_dict = {}
    
    if len(df_with_coords) > 0:
        # 아파트 좌표를 라디안으로 변환
        apt_coords_rad = np.column_stack([
            np.radians(df_with_coords['좌표X'].values),
            np.radians(df_with_coords['좌표Y'].values)
        ])
        
        # 가장 가까운 버스정거장까지의 거리 확인
        print(f"가장 가까운 버스정거장 거리 확인 중...")
        distances_rad, indices = nn_model.kneighbors(apt_coords_rad)
        distances_km = distances_rad.flatten() * R  # km로 변환
        
        # 각 아파트에 대해 서비스권역 여부 확인 (Boolean 값으로 표시)
        for orig_idx, dist_km in zip(df_with_coords.index, distances_km):
            dist_m = dist_km * 1000  # m로 변환
            
            # 핵심서비스권: 300m 이내
            bus_core_service = dist_m <= 300
            # 일반영향권: 300m ~ 500m 이내 (핵심서비스권에 포함되지 않는 경우)
            bus_general_influence = (dist_m > 300) & (dist_m <= 500)
            
            bus_features_dict[orig_idx] = {
                '버스_핵심서비스권': bus_core_service,
                '버스_일반영향권': bus_general_influence
            }
    
    # 좌표가 없는 아파트는 False로 설정
    for orig_idx in df_without_coords.index:
        bus_features_dict[orig_idx] = {
            '버스_핵심서비스권': False,
            '버스_일반영향권': False
        }
    
    # 결과를 DataFrame으로 변환하여 원본에 추가
    bus_features_df = pd.DataFrame([bus_features_dict[idx] for idx in df.index], index=df.index)
    df = pd.concat([df, bus_features_df], axis=1)
    
    # 통계 출력
    bus_core_count = df['버스_핵심서비스권'].sum()
    bus_general_count = df['버스_일반영향권'].sum()
    print(f"버스 핵심서비스권(300m 이내): {bus_core_count}개 ({bus_core_count/len(df)*100:.2f}%)")
    print(f"버스 일반영향권(300m~500m): {bus_general_count}개 ({bus_general_count/len(df)*100:.2f}%)")
    
    return df


def check_subway_zones(df, subway_df):
    """
    각 아파트에서 지하철역 역세권 여부 확인 (Boolean 값으로 표시)
    법령 기준: 1차 역세권(350m 이내), 2차 역세권(350m~500m 이내)
    부동산 시장 기준: 초역세권(300m 이내), 역세권(300m~500m 이내)
    
    Parameters:
    -----------
    df : DataFrame
        아파트 데이터 (좌표X, 좌표Y 컬럼 필요)
    subway_df : DataFrame
        지하철역 데이터 (경도, 위도 컬럼 필요)
    
    Returns:
    --------
    DataFrame : 원본 데이터에 지하철 역세권 여부가 추가된 데이터
    """
    df = df.copy()
    
    # 지하철역 데이터 전처리
    subway_df = subway_df.copy()
    subway_df = subway_df.dropna(subset=['경도', '위도'])
    subway_df['경도'] = pd.to_numeric(subway_df['경도'], errors='coerce')
    subway_df['위도'] = pd.to_numeric(subway_df['위도'], errors='coerce')
    subway_df = subway_df.dropna(subset=['경도', '위도'])
    
    print(f"지하철역 데이터: {len(subway_df)}개")
    print(f"역세권 범위:")
    print(f"  법령 기준: 1차 역세권(350m 이내), 2차 역세권(350m~500m 이내)")
    print(f"  부동산 시장 기준: 초역세권(300m 이내), 역세권(300m~500m 이내)")
    
    # 좌표를 라디안으로 변환 (Haversine 거리를 위해)
    subway_coords_rad = np.column_stack([
        np.radians(subway_df['경도'].values),  # 경도
        np.radians(subway_df['위도'].values)   # 위도
    ])
    
    # NearestNeighbors 모델 생성 및 학습
    print("NearestNeighbors 모델 학습 중...")
    R = 6371.0  # 지구 반지름 (km)
    
    # 가장 가까운 지하철역의 거리를 확인하기 위해 모델 생성
    nn_model = NearestNeighbors(
        n_neighbors=1,  # 가장 가까운 1개 확인
        algorithm='ball_tree',
        metric='haversine'
    )
    nn_model.fit(subway_coords_rad)
    
    # 아파트 좌표 전처리
    df_with_coords = df[df['좌표X'].notna() & df['좌표Y'].notna()].copy()
    df_without_coords = df[df['좌표X'].isna() | df['좌표Y'].isna()].copy()
    
    # 결과 저장용 딕셔너리
    subway_features_dict = {}
    
    if len(df_with_coords) > 0:
        # 아파트 좌표를 라디안으로 변환
        apt_coords_rad = np.column_stack([
            np.radians(df_with_coords['좌표X'].values),
            np.radians(df_with_coords['좌표Y'].values)
        ])
        
        # 가장 가까운 지하철역까지의 거리 확인
        print(f"가장 가까운 지하철역 거리 확인 중...")
        distances_rad, indices = nn_model.kneighbors(apt_coords_rad)
        distances_km = distances_rad.flatten() * R  # km로 변환
        
        # 각 아파트에 대해 역세권 여부 확인 (Boolean 값으로 표시)
        for orig_idx, dist_km in zip(df_with_coords.index, distances_km):
            dist_m = dist_km * 1000  # m로 변환
            
            # 법령 기준
            # 1차 역세권: 350m 이내
            subway_1st_zone = dist_m <= 350
            # 2차 역세권: 350m ~ 500m 이내 (1차 역세권에 포함되지 않는 경우)
            subway_2nd_zone = (dist_m > 350) & (dist_m <= 500)
            
            # 부동산 시장 기준
            # 초역세권: 300m 이내
            subway_premium_zone = dist_m <= 300
            # 역세권: 300m ~ 500m 이내 (초역세권에 포함되지 않는 경우)
            subway_zone = (dist_m > 300) & (dist_m <= 500)
            
            subway_features_dict[orig_idx] = {
                '지하철_1차역세권': subway_1st_zone,
                '지하철_2차역세권': subway_2nd_zone,
                '지하철_초역세권': subway_premium_zone,
                '지하철_역세권': subway_zone
            }
    
    # 좌표가 없는 아파트는 False로 설정
    for orig_idx in df_without_coords.index:
        subway_features_dict[orig_idx] = {
            '지하철_1차역세권': False,
            '지하철_2차역세권': False,
            '지하철_초역세권': False,
            '지하철_역세권': False
        }
    
    # 결과를 DataFrame으로 변환하여 원본에 추가
    subway_features_df = pd.DataFrame([subway_features_dict[idx] for idx in df.index], index=df.index)
    df = pd.concat([df, subway_features_df], axis=1)
    
    # 통계 출력
    subway_1st_count = df['지하철_1차역세권'].sum()
    subway_2nd_count = df['지하철_2차역세권'].sum()
    subway_premium_count = df['지하철_초역세권'].sum()
    subway_zone_count = df['지하철_역세권'].sum()
    
    print(f"지하철 1차 역세권(350m 이내, 법령 기준): {subway_1st_count}개 ({subway_1st_count/len(df)*100:.2f}%)")
    print(f"지하철 2차 역세권(350m~500m, 법령 기준): {subway_2nd_count}개 ({subway_2nd_count/len(df)*100:.2f}%)")
    print(f"지하철 초역세권(300m 이내, 부동산 시장 기준): {subway_premium_count}개 ({subway_premium_count/len(df)*100:.2f}%)")
    print(f"지하철 역세권(300m~500m, 부동산 시장 기준): {subway_zone_count}개 ({subway_zone_count/len(df)*100:.2f}%)")
    
    return df


def add_location_features(df, bus_csv_path, subway_csv_path):
    """
    아파트 데이터에 버스정거장 및 지하철역 역세권 여부 추가
    실거래가에 영향을 미칠 수 있는 역세권 범위를 기준으로 여러 구간 확인
    
    Parameters:
    -----------
    df : DataFrame
        아파트 데이터 (좌표X, 좌표Y 컬럼 필요)
    bus_csv_path : str
        버스정거장 CSV 파일 경로
    subway_csv_path : str
        지하철역 CSV 파일 경로
    
    Returns:
    --------
    DataFrame : 역세권 여부가 추가된 아파트 데이터
    
    추가되는 컬럼 (6개, Boolean 값):
    - 버스_핵심서비스권: 300m 이내
    - 버스_일반영향권: 300m~500m 이내
    - 지하철_1차역세권: 350m 이내 (법령 기준)
    - 지하철_2차역세권: 350m~500m 이내 (법령 기준)
    - 지하철_초역세권: 300m 이내 (부동산 시장 기준)
    - 지하철_역세권: 300m~500m 이내 (부동산 시장 기준)
    """
    print("=" * 60)
    print("역세권 여부 확인 시작")
    print("=" * 60)
    
    # 버스정거장 데이터 로드
    print(f"\n1. 버스정거장 데이터 로딩: {bus_csv_path}")
    bus_df = pd.read_csv(bus_csv_path, encoding='utf-8')
    
    # 지하철역 데이터 로드
    print(f"2. 지하철역 데이터 로딩: {subway_csv_path}")
    subway_df = pd.read_csv(subway_csv_path, encoding='utf-8')
    
    # 버스정거장 서비스권역 여부 확인
    print(f"\n3. 버스정거장 서비스권역 여부 확인 중...")
    df = check_bus_service_zones(df, bus_df)
    
    # 지하철역 역세권 여부 확인
    print(f"\n4. 지하철역 역세권 여부 확인 중...")
    df = check_subway_zones(df, subway_df)
    
    print("\n" + "=" * 60)
    print("역세권 여부 확인 완료!")
    print(f"추가된 컬럼 (총 6개, Boolean):")
    print(f"  버스: '버스_핵심서비스권', '버스_일반영향권'")
    print(f"  지하철: '지하철_1차역세권', '지하철_2차역세권', '지하철_초역세권', '지하철_역세권'")
    print(f"총 컬럼 수: {len(df.columns)}개")
    print("=" * 60)
    
    return df


if __name__ == '__main__':
    # 테스트 코드
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python add_location_features.py <입력_CSV_파일> [출력_CSV_파일]")
        print("예시: python add_location_features.py train.csv train_with_location.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'train_with_location.csv'
    
    # 데이터 로드
    print(f"데이터 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"데이터 shape: {df.shape}")
    
    # 역세권 여부 확인 및 추가
    df_with_location = add_location_features(
        df,
        bus_csv_path='/data/ephemeral/home/py310/bus_feature.csv',
        subway_csv_path='/data/ephemeral/home/py310/subway_feature.csv'
    )
    
    # 결과 저장
    print(f"\n결과 저장: {output_csv}")
    df_with_location.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("완료!")
