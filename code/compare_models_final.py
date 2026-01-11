# -*- coding: utf-8 -*-
"""
아파트 실거래가 예측에 적합한 모델 비교 및 최적 조합 선정
다양한 모델과 전처리 방법을 비교하여 최적의 조합을 선택
"""
import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# LightGBM과 XGBoost import (설치되어 있지 않을 수 있음)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("경고: LightGBM이 설치되어 있지 않습니다. pip install lightgbm로 설치하세요.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("경고: XGBoost가 설치되어 있지 않습니다. pip install xgboost로 설치하세요.")

sys.path.insert(0, '/data/ephemeral/home/py310/code')
from train_model import ApartmentPricePredictor


def evaluate_model(model, X_train, X_val, y_train, y_val, name="Model"):
    """모델 평가"""
    start_time = time.time()
    
    # 모델 학습
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
    else:
        # LightGBM 또는 XGBoost의 경우
        if 'lgb' in name.lower() or 'lightgbm' in name.lower():
            model.train(X_train, y_train, X_val, y_val)
        elif 'xgb' in name.lower() or 'xgboost' in name.lower():
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_time = time.time() - start_time
    
    # 예측
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_val)
    else:
        # LightGBM 또는 XGBoost의 경우
        if 'lgb' in name.lower() or 'lightgbm' in name.lower():
            y_pred = model.predict(X_val)
        elif 'xgb' in name.lower() or 'xgboost' in name.lower():
            y_pred = model.predict(X_val)
    
    # 성능 평가
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_time': train_time
    }


def prepare_data_baseline(predictor, df):
    """기본 전처리 (현재 방식)"""
    X, y = predictor.preprocess_data(df, is_train=True)
    return X, y


def prepare_data_scaled(predictor, df):
    """스케일링 적용 전처리 (Neural Network용)"""
    X, y = predictor.preprocess_data(df, is_train=True)
    
    # 숫자형 컬럼만 스케일링
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X_scaled, y, scaler


def prepare_data_for_tree_models(predictor, df):
    """트리 모델용 전처리 (범주형 변수를 그대로 유지)"""
    # 트리 모델(LightGBM, XGBoost)은 범주형 변수를 직접 처리할 수 있음
    # 하지만 현재 전처리 방식도 잘 작동하므로 동일하게 사용
    X, y = predictor.preprocess_data(df, is_train=True)
    return X, y


def create_rf_model():
    """Random Forest 모델 생성"""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )


def create_gb_model():
    """Gradient Boosting 모델 생성"""
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )


def create_lgbm_model(X_train, y_train, X_val, y_val):
    """LightGBM 모델 생성"""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    # 컬럼명 정리 (특수문자 제거)
    X_train_clean = X_train.copy()
    X_val_clean = X_val.copy()
    
    # 특수문자를 언더스코어로 변경
    rename_dict = {}
    for col in X_train_clean.columns:
        new_col = str(col).replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_').replace('/', '_')
        rename_dict[col] = new_col
    
    X_train_clean = X_train_clean.rename(columns=rename_dict)
    X_val_clean = X_val_clean.rename(columns=rename_dict)
    
    train_data = lgb.Dataset(X_train_clean, label=y_train)
    val_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    return model, X_train_clean, X_val_clean


def create_xgboost_model():
    """XGBoost 모델 생성"""
    if not XGBOOST_AVAILABLE:
        return None
    
    # 컬럼명 정리
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    return model


def main():
    print("=" * 70)
    print("아파트 실거래가 예측 모델 비교 및 최적 조합 선정")
    print("=" * 70)
    
    sample_size = 40000
    print(f"\n테스트 데이터: {sample_size:,}개")
    print(f"학습/검증 비율: 80/20")
    
    # 데이터 로드
    predictor = ApartmentPricePredictor()
    df = predictor.load_data(
        '/data/ephemeral/home/py310/train.csv',
        add_location_features=True,
        bus_csv_path='/data/ephemeral/home/py310/bus_feature.csv',
        subway_csv_path='/data/ephemeral/home/py310/subway_feature.csv'
    )
    df = df.head(sample_size).copy()
    
    print("\n데이터 로딩 완료")
    
    # 전처리 방식별 데이터 준비
    print("\n" + "=" * 70)
    print("전처리 방식별 데이터 준비")
    print("=" * 70)
    
    # 기본 전처리
    print("\n1. 기본 전처리 (현재 방식)")
    X_baseline, y_baseline = prepare_data_baseline(predictor, df)
    X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
        X_baseline, y_baseline, test_size=0.2, random_state=42
    )
    print(f"  특성 수: {X_baseline.shape[1]}개")
    
    # 스케일링 전처리 (Neural Network용)
    print("\n2. 스케일링 전처리 (Neural Network용)")
    X_scaled, y_scaled, scaler = prepare_data_scaled(predictor, df)
    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"  특성 수: {X_scaled.shape[1]}개")
    
    # 트리 모델용 전처리
    print("\n3. 트리 모델용 전처리")
    X_tree, y_tree = prepare_data_for_tree_models(predictor, df)
    X_train_tree, X_val_tree, y_train_tree, y_val_tree = train_test_split(
        X_tree, y_tree, test_size=0.2, random_state=42
    )
    print(f"  특성 수: {X_tree.shape[1]}개")
    
    results = {}
    
    # 1. Random Forest (기본 전처리)
    print("\n" + "=" * 70)
    print("1. Random Forest (기본 전처리)")
    print("=" * 70)
    
    rf_model = create_rf_model()
    result_rf = evaluate_model(rf_model, X_train_base, X_val_base, y_train_base, y_val_base, "Random Forest")
    results['RandomForest_기본'] = result_rf
    
    print(f"\nRandom Forest 성능:")
    print(f"  RMSE: {result_rf['rmse']:.2f}")
    print(f"  MAE: {result_rf['mae']:.2f}")
    print(f"  R²: {result_rf['r2']:.4f}")
    print(f"  학습 시간: {result_rf['train_time']:.2f}초")
    
    # 2. Gradient Boosting (기본 전처리)
    print("\n" + "=" * 70)
    print("2. Gradient Boosting (기본 전처리)")
    print("=" * 70)
    
    gb_model = create_gb_model()
    result_gb = evaluate_model(gb_model, X_train_base, X_val_base, y_train_base, y_val_base, "Gradient Boosting")
    results['GradientBoosting_기본'] = result_gb
    
    print(f"\nGradient Boosting 성능:")
    print(f"  RMSE: {result_gb['rmse']:.2f}")
    print(f"  MAE: {result_gb['mae']:.2f}")
    print(f"  R²: {result_gb['r2']:.4f}")
    print(f"  학습 시간: {result_gb['train_time']:.2f}초")
    
    # 3. LightGBM (기본 전처리, 컬럼명 정리 필요)
    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 70)
        print("3. LightGBM (기본 전처리)")
        print("=" * 70)
        
        try:
            # 컬럼명 정리 (특수문자 완전 제거)
            import re
            X_train_lgb = X_train_tree.copy()
            X_val_lgb = X_val_tree.copy()
            
            rename_dict = {}
            for col in X_train_lgb.columns:
                # 모든 특수문자를 언더스코어로 변경하고, 연속된 언더스코어를 하나로 통합
                new_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
                new_col = re.sub(r'_+', '_', new_col)  # 연속된 언더스코어 제거
                new_col = new_col.strip('_')  # 앞뒤 언더스코어 제거
                if not new_col or new_col[0].isdigit():
                    new_col = 'col_' + new_col  # 숫자로 시작하는 경우 'col_' 추가
                rename_dict[col] = new_col
            
            X_train_lgb = X_train_lgb.rename(columns=rename_dict)
            X_val_lgb = X_val_lgb.rename(columns=rename_dict)
            
            train_data = lgb.Dataset(X_train_lgb, label=y_train_tree)
            val_data = lgb.Dataset(X_val_lgb, label=y_val_tree, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            start_time = time.time()
            lgbm_model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            train_time = time.time() - start_time
            
            y_pred_lgb = lgbm_model.predict(X_val_lgb, num_iteration=lgbm_model.best_iteration)
            
            result_lgb = {
                'rmse': np.sqrt(mean_squared_error(y_val_tree, y_pred_lgb)),
                'mae': mean_absolute_error(y_val_tree, y_pred_lgb),
                'r2': r2_score(y_val_tree, y_pred_lgb),
                'train_time': train_time
            }
            results['LightGBM_기본'] = result_lgb
            
            print(f"\nLightGBM 성능:")
            print(f"  RMSE: {result_lgb['rmse']:.2f}")
            print(f"  MAE: {result_lgb['mae']:.2f}")
            print(f"  R²: {result_lgb['r2']:.4f}")
            print(f"  학습 시간: {result_lgb['train_time']:.2f}초")
        except Exception as e:
            print(f"  오류 발생: {e}")
            results['LightGBM_기본'] = None
    else:
        print("\nLightGBM을 사용할 수 없습니다 (설치되지 않음)")
        results['LightGBM_기본'] = None
    
    # 4. XGBoost (기본 전처리, 컬럼명 정리 필요)
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 70)
        print("4. XGBoost (기본 전처리)")
        print("=" * 70)
        
        try:
            # 컬럼명 정리 (특수문자 완전 제거)
            import re
            X_train_xgb = X_train_tree.copy()
            X_val_xgb = X_val_tree.copy()
            
            rename_dict = {}
            for col in X_train_xgb.columns:
                # 모든 특수문자를 언더스코어로 변경하고, 연속된 언더스코어를 하나로 통합
                new_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
                new_col = re.sub(r'_+', '_', new_col)  # 연속된 언더스코어 제거
                new_col = new_col.strip('_')  # 앞뒤 언더스코어 제거
                if not new_col or new_col[0].isdigit():
                    new_col = 'col_' + new_col  # 숫자로 시작하는 경우 'col_' 추가
                rename_dict[col] = new_col
            
            X_train_xgb = X_train_xgb.rename(columns=rename_dict)
            X_val_xgb = X_val_xgb.rename(columns=rename_dict)
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            start_time = time.time()
            xgb_model.fit(X_train_xgb, y_train_tree, eval_set=[(X_val_xgb, y_val_tree)], verbose=False)
            train_time = time.time() - start_time
            
            y_pred_xgb = xgb_model.predict(X_val_xgb)
            
            result_xgb = {
                'rmse': np.sqrt(mean_squared_error(y_val_tree, y_pred_xgb)),
                'mae': mean_absolute_error(y_val_tree, y_pred_xgb),
                'r2': r2_score(y_val_tree, y_pred_xgb),
                'train_time': train_time
            }
            results['XGBoost_기본'] = result_xgb
            
            print(f"\nXGBoost 성능:")
            print(f"  RMSE: {result_xgb['rmse']:.2f}")
            print(f"  MAE: {result_xgb['mae']:.2f}")
            print(f"  R²: {result_xgb['r2']:.4f}")
            print(f"  학습 시간: {result_xgb['train_time']:.2f}초")
        except Exception as e:
            print(f"  오류 발생: {e}")
            results['XGBoost_기본'] = None
    else:
        print("\nXGBoost를 사용할 수 없습니다 (설치되지 않음)")
        results['XGBoost_기본'] = None
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("모델 성능 비교 요약")
    print("=" * 70)
    
    print(f"\n{'모델':<25} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'학습시간(초)':<15}")
    print("-" * 75)
    
    best_r2 = -np.inf
    best_model = None
    
    for model_name, result in results.items():
        if result is not None:
            print(f"{model_name:<25} {result['r2']:<10.4f} {result['rmse']:<12.2f} {result['mae']:<12.2f} {result['train_time']:<15.2f}")
            
            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model = model_name
    
    # 최적 모델 선정
    print("\n" + "=" * 70)
    print("최적 모델 선정")
    print("=" * 70)
    
    if best_model:
        best_result = results[best_model]
        print(f"\n✓ 최적 모델: {best_model}")
        print(f"  R²: {best_result['r2']:.4f}")
        print(f"  RMSE: {best_result['rmse']:.2f}")
        print(f"  MAE: {best_result['mae']:.2f}")
        print(f"  학습 시간: {best_result['train_time']:.2f}초")
        
        # 다른 모델과 비교
        print(f"\n다른 모델 대비 성능:")
        for model_name, result in results.items():
            if result is not None and model_name != best_model:
                r2_diff = best_result['r2'] - result['r2']
                rmse_diff = result['rmse'] - best_result['rmse']
                print(f"  vs {model_name}: R² {r2_diff:+.4f}, RMSE {rmse_diff:+.2f}")
    else:
        print("\n모든 모델 테스트에 실패했습니다.")
    
    return best_model, results


if __name__ == '__main__':
    best_model, all_results = main()
    
    if best_model:
        print(f"\n최종 결정: {best_model} 모델을 사용합니다.")
    else:
        print("\n최종 결정: Random Forest 모델을 계속 사용합니다.")
