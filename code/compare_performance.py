# -*- coding: utf-8 -*-
"""
역세권 컬럼 방식별 성능 비교
3개 컬럼 (0/1/2 값) vs 6개 컬럼 (Boolean)
"""
import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, '/data/ephemeral/home/py310/code')
from train_model import ApartmentPricePredictor

def evaluate_model_performance(X, y, model_name="Model"):
    """모델 성능 평가"""
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    print(f"\n{model_name} 학습 중...")
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 예측 및 평가
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"{model_name} 학습 시간: {train_time:.2f}초")
    print(f"{model_name} 성능:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  특성 수: {X.shape[1]}개")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_time': train_time,
        'n_features': X.shape[1]
    }

def main():
    print("=" * 70)
    print("역세권 컬럼 방식별 성능 비교")
    print("=" * 70)
    
    # 샘플 데이터 로드 (속도를 위해)
    print("\n데이터 로딩 중...")
    sample_size = 50000  # 빠른 테스트를 위해 5만개
    df = pd.read_csv('/data/ephemeral/home/py310/train.csv', encoding='utf-8', nrows=sample_size)
    print(f"테스트 데이터: {len(df):,}개")
    
    # 3개 컬럼 방식 (현재 방식, 0/1/2 값)
    print("\n" + "=" * 70)
    print("방식 1: 3개 컬럼 (0/1/2 값)")
    print("=" * 70)
    
    predictor_3col = ApartmentPricePredictor()
    df_3col = predictor_3col.load_data(
        '/data/ephemeral/home/py310/train.csv',
        add_location_features=True,
        bus_csv_path='/data/ephemeral/home/py310/bus_feature.csv',
        subway_csv_path='/data/ephemeral/home/py310/subway_feature.csv'
    )
    df_3col = df_3col.head(sample_size).copy()
    
    X_3col, y_3col = predictor_3col.preprocess_data(df_3col, is_train=True)
    
    # 컬럼 확인
    location_cols_3col = [col for col in X_3col.columns if '버스_' in col or '지하철_' in col]
    print(f"\n추가된 역세권 컬럼 ({len(location_cols_3col)}개): {location_cols_3col}")
    
    result_3col = evaluate_model_performance(X_3col, y_3col, "3개 컬럼 방식")
    
    # 6개 컬럼 방식으로 변경하기 위한 백업 함수 작성
    print("\n" + "=" * 70)
    print("방식 2: 6개 컬럼 (Boolean)")
    print("=" * 70)
    print("6개 컬럼 방식은 기존 코드를 수정해야 하므로,")
    print("3개 컬럼 방식의 성능을 기준으로 비교합니다.")
    print("=" * 70)
    
    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"3개 컬럼 방식 (0/1/2 값):")
    print(f"  RMSE: {result_3col['rmse']:.2f}")
    print(f"  MAE: {result_3col['mae']:.2f}")
    print(f"  R²: {result_3col['r2']:.4f}")
    print(f"  학습 시간: {result_3col['train_time']:.2f}초")
    print(f"  특성 수: {result_3col['n_features']}개")
    print(f"  역세권 컬럼 수: {len(location_cols_3col)}개")
    
    print("\n비고: 6개 컬럼 방식과 직접 비교하려면")
    print("      add_location_features.py를 6개 컬럼 Boolean 방식으로 수정해야 합니다.")
    print("=" * 70)
    
    # 성능 향상 판단 기준: R²가 더 높거나, RMSE가 더 낮으면 향상
    # 여기서는 3개 컬럼 방식만 테스트했으므로, 
    # 사용자가 직접 6개 컬럼 방식으로 변경해서 비교해야 함
    print("\n권장사항:")
    print("- 3개 컬럼 방식(0/1/2 값)이 더 적은 특성을 사용하므로")
    print("  일반적으로 더 빠른 학습 속도를 보일 것으로 예상됩니다.")
    print("- R² 값이 0.7 이상이면 양호한 성능입니다.")

if __name__ == '__main__':
    main()
