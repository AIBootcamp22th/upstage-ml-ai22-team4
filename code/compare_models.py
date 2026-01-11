# -*- coding: utf-8 -*-
"""
Random Forest vs LightGBM 성능 비교
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

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("경고: LightGBM이 설치되지 않았습니다. pip install lightgbm으로 설치하세요.")

sys.path.insert(0, '/data/ephemeral/home/py310/code')
from train_model import ApartmentPricePredictor

def evaluate_rf_model(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20, random_state=42):
    """Random Forest 모델 평가"""
    print(f"\nRandom Forest 모델 학습 중...")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
    
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 예측 및 평가
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    return {
        'model': model,
        'train_time': train_time,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

def evaluate_lgbm_model(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20, random_state=42):
    """LightGBM 모델 평가"""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    print(f"\nLightGBM 모델 학습 중...")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
    
    start_time = time.time()
    
    # LightGBM은 특수 문자를 포함한 컬럼 이름을 지원하지 않으므로
    # 컬럼 이름을 변경하여 복사
    X_train_clean = X_train.copy()
    X_val_clean = X_val.copy()
    
    # 컬럼 이름을 숫자 인덱스로 변경 (LightGBM 호환)
    feature_names = [f'feature_{i}' for i in range(X_train_clean.shape[1])]
    X_train_clean.columns = feature_names
    X_val_clean.columns = feature_names
    
    # LightGBM 파라미터 설정 (최적화된 버전)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': min(2**max_depth - 1, 127),  # max_depth에 맞춰 조정
        'learning_rate': 0.01,  # 더 작은 learning rate
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,  # L1 정규화
        'reg_lambda': 0.1,  # L2 정규화
        'verbose': -1,
        'random_state': random_state,
        'max_depth': max_depth,
        'force_col_wise': True  # 컬럼 단위 학습 활성화
    }
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train_clean, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data, feature_name=feature_names)
    
    # 모델 학습 (더 많은 반복과 early stopping)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators * 2,  # learning rate가 작으므로 더 많은 반복
        valid_sets=[train_data, val_data],
        valid_names=['train', 'eval'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)  # 로그 비활성화
        ]
    )
    
    train_time = time.time() - start_time
    
    # 예측 및 평가 (컬럼 이름 변경된 데이터 사용)
    train_pred = model.predict(X_train_clean, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val_clean, num_iteration=model.best_iteration)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    return {
        'model': model,
        'train_time': train_time,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

def main():
    print("=" * 70)
    print("모델 성능 비교: Random Forest vs LightGBM")
    print("=" * 70)
    
    # 데이터 준비
    sample_size = 50000  # 빠른 테스트를 위해 5만개
    print(f"\n테스트 데이터: {sample_size:,}개")
    
    predictor = ApartmentPricePredictor()
    
    # 데이터 로드 및 전처리
    print("\n데이터 로딩 및 전처리 중...")
    df = predictor.load_data(
        '/data/ephemeral/home/py310/train.csv',
        add_location_features=True,
        bus_csv_path='/data/ephemeral/home/py310/bus_feature.csv',
        subway_csv_path='/data/ephemeral/home/py310/subway_feature.csv'
    )
    df = df.head(sample_size).copy()
    
    X, y = predictor.preprocess_data(df, is_train=True)
    print(f"전처리 완료: X.shape={X.shape}, y.shape={y.shape}")
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"데이터 분할: 학습 {len(X_train):,}개, 검증 {len(X_val):,}개")
    
    # Random Forest 평가
    print("\n" + "=" * 70)
    print("1. Random Forest 모델")
    print("=" * 70)
    rf_result = evaluate_rf_model(
        X_train, y_train, X_val, y_val,
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    
    print(f"\n=== Random Forest 학습 데이터 성능 ===")
    print(f"RMSE: {rf_result['train_rmse']:.2f}")
    print(f"MAE: {rf_result['train_mae']:.2f}")
    print(f"R²: {rf_result['train_r2']:.4f}")
    
    print(f"\n=== Random Forest 검증 데이터 성능 ===")
    print(f"RMSE: {rf_result['val_rmse']:.2f}")
    print(f"MAE: {rf_result['val_mae']:.2f}")
    print(f"R²: {rf_result['val_r2']:.4f}")
    print(f"학습 시간: {rf_result['train_time']:.2f}초")
    
    # LightGBM 평가
    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 70)
        print("2. LightGBM 모델")
        print("=" * 70)
        lgbm_result = evaluate_lgbm_model(
            X_train, y_train, X_val, y_val,
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        if lgbm_result:
            print(f"\n=== LightGBM 학습 데이터 성능 ===")
            print(f"RMSE: {lgbm_result['train_rmse']:.2f}")
            print(f"MAE: {lgbm_result['train_mae']:.2f}")
            print(f"R²: {lgbm_result['train_r2']:.4f}")
            
            print(f"\n=== LightGBM 검증 데이터 성능 ===")
            print(f"RMSE: {lgbm_result['val_rmse']:.2f}")
            print(f"MAE: {lgbm_result['val_mae']:.2f}")
            print(f"R²: {lgbm_result['val_r2']:.4f}")
            print(f"학습 시간: {lgbm_result['train_time']:.2f}초")
            
            # 결과 비교
            print("\n" + "=" * 70)
            print("성능 비교 결과")
            print("=" * 70)
            
            # 검증 데이터 기준 비교
            r2_diff = lgbm_result['val_r2'] - rf_result['val_r2']
            rmse_diff = rf_result['val_rmse'] - lgbm_result['val_rmse']
            mae_diff = rf_result['val_mae'] - lgbm_result['val_mae']
            time_diff = lgbm_result['train_time'] - rf_result['train_time']
            
            print(f"\n검증 데이터 기준:")
            print(f"  R² 차이: {r2_diff:+.4f} (LightGBM이 {'더 좋음' if r2_diff > 0 else '더 나쁨'})")
            print(f"  RMSE 차이: {rmse_diff:+.2f} (LightGBM이 {'더 좋음' if rmse_diff > 0 else '더 나쁨'})")
            print(f"  MAE 차이: {mae_diff:+.2f} (LightGBM이 {'더 좋음' if mae_diff > 0 else '더 나쁨'})")
            print(f"  학습 시간 차이: {time_diff:+.2f}초 (LightGBM이 {'더 느림' if time_diff > 0 else '더 빠름'})")
            
            # 결론
            print("\n" + "=" * 70)
            print("결론")
            print("=" * 70)
            
            # R²가 더 높거나, R²가 비슷한데 RMSE가 더 낮으면 LightGBM 승리
            if r2_diff > 0.001 or (abs(r2_diff) <= 0.001 and rmse_diff > 0):
                print("✓ LightGBM이 더 좋은 성능을 보입니다.")
                print("  → LightGBM 모델로 변경합니다.")
                better_model = 'lgbm'
            elif r2_diff < -0.001 or (abs(r2_diff) <= 0.001 and rmse_diff < 0):
                print("✓ Random Forest가 더 좋은 성능을 보입니다.")
                print("  → Random Forest 모델을 유지합니다.")
                better_model = 'rf'
            else:
                print("두 모델의 성능이 비슷합니다.")
                print("  학습 시간을 고려하면: ", end="")
                if time_diff < 0:
                    print("LightGBM이 더 빠르므로 LightGBM을 선택합니다.")
                    better_model = 'lgbm'
                else:
                    print("Random Forest가 더 빠르므로 Random Forest를 유지합니다.")
                    better_model = 'rf'
            
            print("=" * 70)
            
            return better_model, rf_result, lgbm_result
        else:
            print("LightGBM 모델 학습에 실패했습니다.")
            return 'rf', rf_result, None
    else:
        print("\n경고: LightGBM이 설치되지 않아 비교할 수 없습니다.")
        print("  pip install lightgbm으로 설치하세요.")
        return 'rf', rf_result, None

if __name__ == '__main__':
    better_model, rf_result, lgbm_result = main()
    
    if better_model == 'lgbm':
        print(f"\n최종 결정: LightGBM 모델 사용 (R²={lgbm_result['val_r2']:.4f}, RMSE={lgbm_result['val_rmse']:.2f})")
    else:
        print(f"\n최종 결정: Random Forest 모델 유지 (R²={rf_result['val_r2']:.4f}, RMSE={rf_result['val_rmse']:.2f})")
