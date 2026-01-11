# -*- coding: utf-8 -*-
"""
층수 관련 파생변수 생성
지역별 층수에 따른 실거래가 변동 패턴 반영
"""
import pandas as pd
import numpy as np


def add_floor_features(df):
    """
    층수 관련 파생변수 생성
    
    Parameters:
    -----------
    df : DataFrame
        아파트 데이터 (층, 시군구 컬럼 필요)
    
    Returns:
    --------
    DataFrame : 층수 관련 파생변수가 추가된 데이터
    
    추가되는 컬럼:
    - 층수_구간: 층수 구간 카테고리 (저층/중저층/중층/고층/초고층)
    - 층수_코드: 층수 구간 숫자 인코딩 (0~4)
    - 층수_저층, 층수_중저층, 층수_중층, 층수_고층, 층수_초고층: 더미 변수
    - 층수_제곱: 층수 제곱 (고층 프리미엄 반영)
    - 층수_고층여부: 21층 이상 여부
    - 층수_초고층여부: 50층 이상 여부
    - 층수_지역정규화: 시군구별 평균 층수 대비 상대적 층수
    """
    df = df.copy()
    
    if '층' not in df.columns:
        print("경고: '층' 컬럼이 없습니다. 층수 파생변수를 생성할 수 없습니다.")
        return df
    
    print("층수 관련 파생변수 생성 중...")
    
    # 층수 구간 분류
    def categorize_floor(floor):
        """층수를 구간별로 분류"""
        if pd.isna(floor):
            return 'Unknown'
        elif floor <= 5:
            return '저층_1_5'
        elif floor <= 10:
            return '중저층_6_10'
        elif floor <= 20:
            return '중층_11_20'
        elif floor <= 50:
            return '고층_21_50'
        else:
            return '초고층_50+'
    
    df['층수_구간'] = df['층'].apply(categorize_floor)
    
    # 층수 구간 숫자 인코딩
    floor_mapping = {
        'Unknown': 0,
        '저층_1_5': 1,
        '중저층_6_10': 2,
        '중층_11_20': 3,
        '고층_21_50': 4,
        '초고층_50+': 5
    }
    df['층수_코드'] = df['층수_구간'].map(floor_mapping).fillna(0).astype(int)
    
    # 층수 구간별 더미 변수 생성
    df['층수_저층'] = (df['층수_코드'] == 1).astype(int)
    df['층수_중저층'] = (df['층수_코드'] == 2).astype(int)
    df['층수_중층'] = (df['층수_코드'] == 3).astype(int)
    df['층수_고층'] = (df['층수_코드'] == 4).astype(int)
    df['층수_초고층'] = (df['층수_코드'] == 5).astype(int)
    
    # 층수 제곱 (고층 프리미엄 반영)
    df['층수_제곱'] = df['층'] ** 2
    
    # 고층 여부 (21층 이상)
    df['층수_고층여부'] = (df['층'] >= 21).astype(int)
    
    # 초고층 여부 (50층 이상)
    df['층수_초고층여부'] = (df['층'] >= 50).astype(int)
    
    # 시군구별 평균 층수 대비 상대적 층수 (지역별 정규화)
    if '시군구' in df.columns:
        # 시군구별 평균 층수 계산 (전체 데이터 기준)
        region_mean_floor = df.groupby('시군구')['층'].mean().to_dict()
        
        # 각 시군구의 평균 층수 대비 상대적 층수 계산
        df['시군구_평균층수'] = df['시군구'].map(region_mean_floor).fillna(df['층'].mean())
        df['층수_지역정규화'] = (df['층'] - df['시군구_평균층수']) / (df['시군구_평균층수'] + 1e-6)
        
        # 시군구_평균층수는 임시 컬럼이므로 제거
        df = df.drop(columns=['시군구_평균층수'])
        
        print(f"  시군구별 층수 정규화 완료 (시군구 수: {len(region_mean_floor)}개)")
    else:
        print("  경고: '시군구' 컬럼이 없어 지역별 층수 정규화를 수행하지 않습니다.")
        df['층수_지역정규화'] = 0
    
    # 통계 출력
    print(f"  층수 구간 분포:")
    floor_dist = df['층수_구간'].value_counts().sort_index()
    for floor_cat, count in floor_dist.items():
        pct = count / len(df) * 100
        print(f"    {floor_cat}: {count:,}개 ({pct:.2f}%)")
    
    print(f"  층수 파생변수 생성 완료 (추가 컬럼 수: 12개)")
    
    return df


if __name__ == '__main__':
    # 테스트 코드
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python add_floor_features.py <입력_CSV_파일> [출력_CSV_파일]")
        print("예시: python add_floor_features.py train.csv train_with_floor.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'train_with_floor.csv'
    
    # 데이터 로드
    print(f"데이터 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8', nrows=10000)  # 테스트용
    print(f"데이터 shape: {df.shape}")
    
    # 층수 파생변수 추가
    df_with_floor = add_floor_features(df)
    
    # 결과 저장
    print(f"\n결과 저장: {output_csv}")
    df_with_floor.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("완료!")
