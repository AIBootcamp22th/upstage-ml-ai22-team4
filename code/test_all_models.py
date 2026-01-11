# -*- coding: utf-8 -*-
"""
모든 모델 테스트 및 최적 성능 비교
LightGBM, XGBoost 포함하여 모든 모델 테스트
"""
import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor,
    AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import time
import warnings
import re
warnings.filterwarnings('ignore')

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# LightGBM과 XGBoost import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("경고: LightGBM이 설치되어 있지 않습니다.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("경고: XGBoost가 설치되어 있지 않습니다.")

sys.path.insert(0, '/data/ephemeral/home/py310/code')
from train_model import ApartmentPricePredictor


def clean_column_names(df):
    """컬럼명 정리: 특수문자 제거 및 안전한 이름으로 변경"""
    df_clean = df.copy()
    rename_dict = {}
    
    for col in df_clean.columns:
        # 모든 특수문자를 언더스코어로 변경
        new_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        # 연속된 언더스코어를 하나로 통합
        new_col = re.sub(r'_+', '_', new_col)
        # 앞뒤 언더스코어 제거
        new_col = new_col.strip('_')
        # 비어있거나 숫자로 시작하는 경우 처리
        if not new_col:
            new_col = 'col_' + str(hash(col) % 10000)
        elif new_col[0].isdigit():
            new_col = 'col_' + new_col
        
        # 중복 방지: 이미 존재하는 경우 인덱스 추가
        if new_col in rename_dict.values():
            new_col = new_col + '_' + str(len([v for v in rename_dict.values() if v.startswith(new_col)]))
        
        rename_dict[col] = new_col
    
    df_clean = df_clean.rename(columns=rename_dict)
    return df_clean, rename_dict


def prepare_data_for_models(predictor, df, model_type='random_forest'):
    """모델별 최적 전처리 적용"""
    # 기본 전처리 (Target Encoding, Frequency Encoding, 연식 변수 등)
    X, y = predictor.preprocess_data(df, is_train=True)
    
    if model_type in ['lightgbm', 'lgbm', 'lgb']:
        # LightGBM: 컬럼명 정리 필요
        X_clean, rename_dict = clean_column_names(X)
        return X_clean, y, rename_dict
    elif model_type in ['xgboost', 'xgb']:
        # XGBoost: 컬럼명 정리 필요
        X_clean, rename_dict = clean_column_names(X)
        return X_clean, y, rename_dict
    else:
        # Random Forest, Gradient Boosting: 현재 전처리 유지
        return X, y


def evaluate_rf_model(X_train, X_val, y_train, y_val, config):
    """Random Forest 모델 평가"""
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config.get('min_samples_split', 5),
        min_samples_leaf=config.get('min_samples_leaf', 2),
        max_features=config.get('max_features', 'sqrt'),
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_val)
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
        'mae': mean_absolute_error(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
        'train_time': train_time,
        'model': model
    }


def evaluate_gb_model(X_train, X_val, y_train, y_val, config):
    """Gradient Boosting 모델 평가"""
    model = GradientBoostingRegressor(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        min_samples_split=config.get('min_samples_split', 5),
        min_samples_leaf=config.get('min_samples_leaf', 2),
        subsample=config.get('subsample', 0.8),
        random_state=42,
        verbose=0
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_val)
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
        'mae': mean_absolute_error(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
        'train_time': train_time,
        'model': model
    }


def evaluate_lgbm_model(X_train, X_val, y_train, y_val, config, rename_dict=None):
    """LightGBM 모델 평가 (컬럼명 정리 포함)"""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    try:
        # 범주형 변수 식별 (원본 컬럼명 기준)
        categorical_features = []
        if rename_dict:
            # 역매핑: 정리된 컬럼명 -> 원본 컬럼명
            reverse_dict = {v: k for k, v in rename_dict.items()}
            
            # 범주형 변수 찾기 (원본 컬럼명에 범주형 특징이 있는 경우)
            # Label Encoding이 적용되었지만, 원래 범주형이었던 변수 식별
            # 실제로는 이미 숫자로 인코딩되어 있으므로 범주형으로 지정하지 않음
            pass
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config['learning_rate'],
            'feature_fraction': config.get('feature_fraction', 0.8),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'min_data_in_leaf': config.get('min_data_in_leaf', 20),
            'min_sum_hessian_in_leaf': config.get('min_sum_hessian_in_leaf', 1e-3),
            'verbose': -1,
            'random_state': 42
        }
        
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            num_boost_round=config['n_estimators'],
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=config.get('early_stopping_rounds', 20), verbose=False)]
        )
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'train_time': train_time,
            'model': model,
            'best_iteration': model.best_iteration
        }
    except Exception as e:
        print(f"  오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_xgb_model(X_train, X_val, y_train, y_val, config, rename_dict=None):
    """XGBoost 모델 평가 (컬럼명 정리 포함)"""
    if not XGBOOST_AVAILABLE:
        return None
    
    try:
        # XGBoost는 DataFrame을 직접 받을 수 있지만, 안전을 위해 numpy array로 변환
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            min_child_weight=config.get('min_child_weight', 3),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            gamma=config.get('gamma', 0),
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method='hist'  # 메모리 효율성
        )
        
        # XGBoost 버전별 호환성 처리
        # XGBoost 2.0+ 버전에서는 early_stopping_rounds 대신 다른 방법 사용
        start_time = time.time()
        
        # eval_set과 verbose 설정
        fit_params = {
            'eval_set': [(X_val_array, y_val)],
            'verbose': False
        }
        
        # early_stopping_rounds는 XGBRegressor 생성자에 설정하거나
        # 더 최신 버전에서는 fit()에 전달 가능
        try:
            model.fit(X_train_array, y_train, **fit_params, early_stopping_rounds=config.get('early_stopping_rounds', 20))
        except TypeError:
            # early_stopping_rounds가 지원되지 않는 경우 제거
            try:
                model.fit(X_train_array, y_train, **fit_params)
            except Exception as e2:
                # eval_set도 문제가 있는 경우
                model.fit(X_train_array, y_train, verbose=False)
        
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_val_array)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'train_time': train_time,
            'model': model
        }
    except Exception as e:
        print(f"  오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("모든 모델 테스트 및 최적 성능 비교")
    print("=" * 70)
    
    sample_size = 50000
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
    
    all_results = {}
    
    # 1. Random Forest (다양한 파라미터)
    print("\n" + "=" * 70)
    print("1. Random Forest (최적 파라미터 탐색)")
    print("=" * 70)
    
    X_rf, y_rf = prepare_data_for_models(predictor, df, 'random_forest')
    X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42
    )
    
    rf_configs = [
        {'name': 'RF_기본', 'n_estimators': 100, 'max_depth': 20},
        {'name': 'RF_최적1', 'n_estimators': 150, 'max_depth': 25},
        {'name': 'RF_최적2', 'n_estimators': 200, 'max_depth': 25},
        {'name': 'RF_튜닝1', 'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 3},
    ]
    
    for config in rf_configs:
        print(f"\n{config['name']} 테스트 중...")
        result = evaluate_rf_model(X_train_rf, X_val_rf, y_train_rf, y_val_rf, config)
        all_results[config['name']] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    
    # 2. Gradient Boosting (다양한 파라미터)
    print("\n" + "=" * 70)
    print("2. Gradient Boosting (최적 파라미터 탐색)")
    print("=" * 70)
    
    X_gb, y_gb = prepare_data_for_models(predictor, df, 'gradient_boosting')
    X_train_gb, X_val_gb, y_train_gb, y_val_gb = train_test_split(
        X_gb, y_gb, test_size=0.2, random_state=42
    )
    
    gb_configs = [
        {'name': 'GB_기본', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
        {'name': 'GB_최적1', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7},
        {'name': 'GB_최적2', 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 7},
        {'name': 'GB_튜닝1', 'n_estimators': 250, 'learning_rate': 0.04, 'max_depth': 8, 'subsample': 0.85},
    ]
    
    for config in gb_configs:
        print(f"\n{config['name']} 테스트 중...")
        result = evaluate_gb_model(X_train_gb, X_val_gb, y_train_gb, y_val_gb, config)
        all_results[config['name']] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    
    # 3. LightGBM (컬럼명 정리 및 최적 파라미터)
    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 70)
        print("3. LightGBM (컬럼명 정리 및 최적 파라미터 탐색)")
        print("=" * 70)
        
        X_lgb, y_lgb, rename_dict_lgb = prepare_data_for_models(predictor, df, 'lightgbm')
        X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
            X_lgb, y_lgb, test_size=0.2, random_state=42
        )
        
        print(f"컬럼명 정리 완료: {len(X_lgb.columns)}개 컬럼")
        
        lgb_configs = [
            {'name': 'LGBM_기본', 'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31},
            {'name': 'LGBM_최적1', 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31},
            {'name': 'LGBM_최적2', 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 31},
            {'name': 'LGBM_튜닝1', 'n_estimators': 250, 'learning_rate': 0.04, 'num_leaves': 50, 'min_data_in_leaf': 15},
        ]
        
        for config in lgb_configs:
            print(f"\n{config['name']} 테스트 중...")
            result = evaluate_lgbm_model(X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb, config, rename_dict_lgb)
            if result:
                all_results[config['name']] = result
                print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
            else:
                print(f"  테스트 실패")
    else:
        print("\nLightGBM을 사용할 수 없습니다 (설치되지 않음)")
    
    # 4. XGBoost (컬럼명 정리 및 최적 파라미터)
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 70)
        print("4. XGBoost (컬럼명 정리 및 최적 파라미터 탐색)")
        print("=" * 70)
        
        X_xgb, y_xgb, rename_dict_xgb = prepare_data_for_models(predictor, df, 'xgboost')
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_xgb, y_xgb, test_size=0.2, random_state=42
        )
        
        print(f"컬럼명 정리 완료: {len(X_xgb.columns)}개 컬럼")
        
        xgb_configs = [
            {'name': 'XGB_기본', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            {'name': 'XGB_최적1', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7},
            {'name': 'XGB_최적2', 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 7},
            {'name': 'XGB_튜닝1', 'n_estimators': 250, 'learning_rate': 0.04, 'max_depth': 8, 'min_child_weight': 2},
        ]
        
        for config in xgb_configs:
            print(f"\n{config['name']} 테스트 중...")
            result = evaluate_xgb_model(X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb, config, rename_dict_xgb)
            if result:
                all_results[config['name']] = result
                print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
            else:
                print(f"  테스트 실패")
    else:
        print("\nXGBoost를 사용할 수 없습니다 (설치되지 않음)")
    
    # 5. 앙상블 모델 테스트
    print("\n" + "=" * 70)
    print("5. 앙상블 모델 테스트")
    print("=" * 70)
    
    # 개별 모델 학습 (최적 파라미터 사용)
    print("\n개별 모델 학습 중 (최적 파라미터 사용)...")
    
    # 공통 데이터 사용 (Random Forest용 데이터셋)
    X_common, y_common = prepare_data_for_models(predictor, df, 'random_forest')
    X_train_common, X_val_common, y_train_common, y_val_common = train_test_split(
        X_common, y_common, test_size=0.2, random_state=42
    )
    
    base_models = []
    base_model_results = {}
    
    # Random Forest (최적)
    print("  Random Forest 학습 중...")
    rf_optimal = RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    start_time = time.time()
    rf_optimal.fit(X_train_common, y_train_common)
    rf_time = time.time() - start_time
    rf_pred = rf_optimal.predict(X_val_common)
    base_models.append(('rf', rf_optimal))
    base_model_results['rf'] = {
        'r2': r2_score(y_val_common, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_val_common, rf_pred)),
        'mae': mean_absolute_error(y_val_common, rf_pred),
        'time': rf_time
    }
    
    # Gradient Boosting (최적)
    print("  Gradient Boosting 학습 중...")
    gb_optimal = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=42,
        verbose=0
    )
    start_time = time.time()
    gb_optimal.fit(X_train_common, y_train_common)
    gb_time = time.time() - start_time
    gb_pred = gb_optimal.predict(X_val_common)
    base_models.append(('gb', gb_optimal))
    base_model_results['gb'] = {
        'r2': r2_score(y_val_common, gb_pred),
        'rmse': np.sqrt(mean_squared_error(y_val_common, gb_pred)),
        'mae': mean_absolute_error(y_val_common, gb_pred),
        'time': gb_time
    }
    
    # XGBoost (최적, 사용 가능한 경우)
    xgb_optimal = None
    if XGBOOST_AVAILABLE:
        print("  XGBoost 학습 중...")
        X_xgb_common, y_xgb_common, _ = prepare_data_for_models(predictor, df, 'xgboost')
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_xgb_common, y_xgb_common, test_size=0.2, random_state=42
        )
        X_train_array = X_train_xgb.values if isinstance(X_train_xgb, pd.DataFrame) else X_train_xgb
        X_val_array = X_val_xgb.values if isinstance(X_val_xgb, pd.DataFrame) else X_val_xgb
        
        xgb_optimal = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=8,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method='hist'
        )
        start_time = time.time()
        xgb_optimal.fit(X_train_array, y_train_xgb, verbose=False)
        xgb_time = time.time() - start_time
        xgb_pred = xgb_optimal.predict(X_val_array)
        
        # XGBoost 결과를 공통 검증 데이터셋으로 변환 필요
        # 간단한 앙상블을 위해 RF/GB만 사용하는 것이 안전
        base_model_results['xgb'] = {
            'r2': r2_score(y_val_xgb, xgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_val_xgb, xgb_pred)),
            'mae': mean_absolute_error(y_val_xgb, xgb_pred),
            'time': xgb_time,
            'predictions': xgb_pred,
            'X_val': X_val_array,
            'y_val': y_val_xgb
        }
    
    # 5-1. VotingRegressor (평균 앙상블)
    print("\n5-1. VotingRegressor (평균 앙상블) 테스트 중...")
    try:
        voting_models = [('rf', rf_optimal), ('gb', gb_optimal)]
        if xgb_optimal is not None:
            # XGBoost는 다른 데이터셋이므로 제외 (또는 별도 처리 필요)
            pass
        
        voting_ensemble = VotingRegressor(estimators=voting_models, weights=None)
        start_time = time.time()
        voting_ensemble.fit(X_train_common, y_train_common)
        voting_time = time.time() - start_time
        voting_pred = voting_ensemble.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, voting_pred)),
            'mae': mean_absolute_error(y_val_common, voting_pred),
            'r2': r2_score(y_val_common, voting_pred),
            'train_time': voting_time,
            'model': voting_ensemble
        }
        all_results['ENS_Voting_RF+GB'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-2. Weighted Average Ensemble (성능 기반 가중 평균)
    print("\n5-2. Weighted Average Ensemble (성능 기반 가중 평균) 테스트 중...")
    try:
        # R² 점수를 기준으로 가중치 계산
        weights = []
        predictions = []
        
        # RF 예측
        rf_pred = rf_optimal.predict(X_val_common)
        rf_weight = base_model_results['rf']['r2']
        weights.append(rf_weight)
        predictions.append(rf_pred)
        
        # GB 예측
        gb_pred = gb_optimal.predict(X_val_common)
        gb_weight = base_model_results['gb']['r2']
        weights.append(gb_weight)
        predictions.append(gb_pred)
        
        # 가중치 정규화 (합이 1이 되도록)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 가중 평균 예측
        weighted_pred = np.zeros(len(y_val_common), dtype=np.float64)
        for pred, weight in zip(predictions, weights):
            weighted_pred += weight * pred.astype(np.float64)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, weighted_pred)),
            'mae': mean_absolute_error(y_val_common, weighted_pred),
            'r2': r2_score(y_val_common, weighted_pred),
            'train_time': rf_time + gb_time,  # 개별 모델 학습 시간 합계
            'weights': dict(zip(['RF', 'GB'], weights)),
            'model': 'WeightedAverage'
        }
        all_results['ENS_Weighted_RF+GB'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
        print(f"  가중치: RF={weights[0]:.3f}, GB={weights[1]:.3f}")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-3. Simple Average Ensemble (단순 평균)
    print("\n5-3. Simple Average Ensemble (단순 평균) 테스트 중...")
    try:
        predictions = [rf_pred, gb_pred]
        avg_pred = np.mean(predictions, axis=0)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, avg_pred)),
            'mae': mean_absolute_error(y_val_common, avg_pred),
            'r2': r2_score(y_val_common, avg_pred),
            'train_time': rf_time + gb_time,
            'model': 'SimpleAverage'
        }
        all_results['ENS_Simple_RF+GB'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-4. StackingRegressor (메타 모델: RidgeCV)
    print("\n5-4. StackingRegressor (메타 모델: RidgeCV) 테스트 중...")
    try:
        base_estimators = [('rf', rf_optimal), ('gb', gb_optimal)]
        stacking_ensemble = StackingRegressor(
            estimators=base_estimators,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1
        )
        start_time = time.time()
        stacking_ensemble.fit(X_train_common, y_train_common)
        stacking_time = time.time() - start_time
        stacking_pred = stacking_ensemble.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, stacking_pred)),
            'mae': mean_absolute_error(y_val_common, stacking_pred),
            'r2': r2_score(y_val_common, stacking_pred),
            'train_time': stacking_time,
            'model': stacking_ensemble
        }
        all_results['ENS_Stacking_RidgeCV'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
        import traceback
        traceback.print_exc()
    
    # 5-5. StackingRegressor (메타 모델: LassoCV)
    print("\n5-5. StackingRegressor (메타 모델: LassoCV) 테스트 중...")
    try:
        base_estimators = [('rf', rf_optimal), ('gb', gb_optimal)]
        stacking_lasso = StackingRegressor(
            estimators=base_estimators,
            final_estimator=LassoCV(cv=5, random_state=42),
            cv=5,
            n_jobs=-1
        )
        start_time = time.time()
        stacking_lasso.fit(X_train_common, y_train_common)
        stacking_lasso_time = time.time() - start_time
        stacking_lasso_pred = stacking_lasso.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, stacking_lasso_pred)),
            'mae': mean_absolute_error(y_val_common, stacking_lasso_pred),
            'r2': r2_score(y_val_common, stacking_lasso_pred),
            'train_time': stacking_lasso_time,
            'model': stacking_lasso
        }
        all_results['ENS_Stacking_LassoCV'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-6. StackingRegressor (메타 모델: ElasticNetCV)
    print("\n5-6. StackingRegressor (메타 모델: ElasticNetCV) 테스트 중...")
    try:
        base_estimators = [('rf', rf_optimal), ('gb', gb_optimal)]
        stacking_elastic = StackingRegressor(
            estimators=base_estimators,
            final_estimator=ElasticNetCV(cv=5, random_state=42),
            cv=5,
            n_jobs=-1
        )
        start_time = time.time()
        stacking_elastic.fit(X_train_common, y_train_common)
        stacking_elastic_time = time.time() - start_time
        stacking_elastic_pred = stacking_elastic.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, stacking_elastic_pred)),
            'mae': mean_absolute_error(y_val_common, stacking_elastic_pred),
            'r2': r2_score(y_val_common, stacking_elastic_pred),
            'train_time': stacking_elastic_time,
            'model': stacking_elastic
        }
        all_results['ENS_Stacking_ElasticNetCV'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-7. StackingRegressor (메타 모델: LinearRegression)
    print("\n5-7. StackingRegressor (메타 모델: LinearRegression) 테스트 중...")
    try:
        base_estimators = [('rf', rf_optimal), ('gb', gb_optimal)]
        stacking_linear = StackingRegressor(
            estimators=base_estimators,
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1
        )
        start_time = time.time()
        stacking_linear.fit(X_train_common, y_train_common)
        stacking_linear_time = time.time() - start_time
        stacking_linear_pred = stacking_linear.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, stacking_linear_pred)),
            'mae': mean_absolute_error(y_val_common, stacking_linear_pred),
            'r2': r2_score(y_val_common, stacking_linear_pred),
            'train_time': stacking_linear_time,
            'model': stacking_linear
        }
        all_results['ENS_Stacking_Linear'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-8. Blending (홀드아웃 세트에서 메타 모델 학습)
    print("\n5-8. Blending (홀드아웃 세트에서 메타 모델 학습) 테스트 중...")
    try:
        # 학습 데이터를 기본 모델 학습용과 메타 모델 학습용으로 분할
        X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
            X_train_common, y_train_common, test_size=0.3, random_state=42
        )
        
        # 기본 모델들 학습
        rf_blend = RandomForestRegressor(
            n_estimators=200, max_depth=30, min_samples_split=3,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1, verbose=0
        )
        rf_blend.fit(X_train_base, y_train_base)
        
        gb_blend = GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=8,
            min_samples_split=5, min_samples_leaf=2, subsample=0.85,
            random_state=42, verbose=0
        )
        gb_blend.fit(X_train_base, y_train_base)
        
        # 메타 데이터 생성 (기본 모델들의 예측값)
        meta_features = np.column_stack([
            rf_blend.predict(X_train_meta),
            gb_blend.predict(X_train_meta)
        ])
        
        # 메타 모델 학습 (RidgeCV)
        meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        start_time = time.time()
        meta_model.fit(meta_features, y_train_meta)
        
        # 검증 데이터에 대한 예측
        val_meta_features = np.column_stack([
            rf_blend.predict(X_val_common),
            gb_blend.predict(X_val_common)
        ])
        blending_pred = meta_model.predict(val_meta_features)
        blending_time = time.time() - start_time + (rf_time + gb_time) * 0.7  # 기본 모델 학습 시간 포함
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, blending_pred)),
            'mae': mean_absolute_error(y_val_common, blending_pred),
            'r2': r2_score(y_val_common, blending_pred),
            'train_time': blending_time,
            'model': {'rf': rf_blend, 'gb': gb_blend, 'meta': meta_model}
        }
        all_results['ENS_Blending'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
        import traceback
        traceback.print_exc()
    
    # 5-9. AdaBoost Regressor (약한 학습기 앙상블)
    print("\n5-9. AdaBoost Regressor (약한 학습기 앙상블) 테스트 중...")
    try:
        base_estimator = DecisionTreeRegressor(max_depth=5, random_state=42)
        adaboost = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        start_time = time.time()
        adaboost.fit(X_train_common, y_train_common)
        adaboost_time = time.time() - start_time
        adaboost_pred = adaboost.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, adaboost_pred)),
            'mae': mean_absolute_error(y_val_common, adaboost_pred),
            'r2': r2_score(y_val_common, adaboost_pred),
            'train_time': adaboost_time,
            'model': adaboost
        }
        all_results['ENS_AdaBoost'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-10. ExtraTrees Regressor (Random Forest 변형)
    print("\n5-10. ExtraTrees Regressor (Random Forest 변형) 테스트 중...")
    try:
        extra_trees = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        start_time = time.time()
        extra_trees.fit(X_train_common, y_train_common)
        extra_trees_time = time.time() - start_time
        extra_trees_pred = extra_trees.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, extra_trees_pred)),
            'mae': mean_absolute_error(y_val_common, extra_trees_pred),
            'r2': r2_score(y_val_common, extra_trees_pred),
            'train_time': extra_trees_time,
            'model': extra_trees
        }
        all_results['ENS_ExtraTrees'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 5-11. Bagging Regressor (Bootstrap Aggregating)
    print("\n5-11. Bagging Regressor (Bootstrap Aggregating) 테스트 중...")
    try:
        base_estimator = DecisionTreeRegressor(max_depth=20, random_state=42)
        bagging = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        start_time = time.time()
        bagging.fit(X_train_common, y_train_common)
        bagging_time = time.time() - start_time
        bagging_pred = bagging.predict(X_val_common)
        
        result = {
            'rmse': np.sqrt(mean_squared_error(y_val_common, bagging_pred)),
            'mae': mean_absolute_error(y_val_common, bagging_pred),
            'r2': r2_score(y_val_common, bagging_pred),
            'train_time': bagging_time,
            'model': bagging
        }
        all_results['ENS_Bagging'] = result
        print(f"  R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}, 시간: {result['train_time']:.2f}초")
    except Exception as e:
        print(f"  오류: {e}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("최종 성능 비교 요약")
    print("=" * 70)
    
    print(f"\n{'모델':<25} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'학습시간(초)':<15}")
    print("-" * 75)
    
    best_r2 = -np.inf
    best_model_name = None
    
    for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True):
        if result:
            print(f"{model_name:<25} {result['r2']:<10.4f} {result['rmse']:<12.2f} {result['mae']:<12.2f} {result['train_time']:<15.2f}")
            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model_name = model_name
    
    # 최적 모델 선정
    print("\n" + "=" * 70)
    print("최적 모델 선정")
    print("=" * 70)
    
    if best_model_name:
        best_result = all_results[best_model_name]
        print(f"\n✓ 최고 성능 모델: {best_model_name}")
        print(f"  R²: {best_result['r2']:.4f}")
        print(f"  RMSE: {best_result['rmse']:.2f}")
        print(f"  MAE: {best_result['mae']:.2f}")
        print(f"  학습 시간: {best_result['train_time']:.2f}초")
    
    # 모델별 최고 성능 비교
    print("\n모델별 최고 성능:")
    model_groups = {
        'Random Forest': [k for k in all_results.keys() if 'RF' in k and 'ENS' not in k],
        'Gradient Boosting': [k for k in all_results.keys() if 'GB' in k and 'ENS' not in k],
        'LightGBM': [k for k in all_results.keys() if 'LGBM' in k],
        'XGBoost': [k for k in all_results.keys() if 'XGB' in k],
        '앙상블': [k for k in all_results.keys() if 'ENS' in k],
    }
    
    best_by_type = {}
    for model_type, model_list in model_groups.items():
        if model_list:
            valid_results = [(k, all_results[k]) for k in model_list if all_results.get(k) and isinstance(all_results[k], dict) and 'r2' in all_results[k]]
            if valid_results:
                best_config = max(valid_results, key=lambda x: x[1]['r2'])[0]
                best_by_type[model_type] = {
                    'name': best_config,
                    'result': all_results[best_config]
                }
                result = all_results[best_config]
                print(f"  {model_type}: {best_config} - R² {result['r2']:.4f}, RMSE {result['rmse']:.2f}, 시간 {result['train_time']:.2f}초")
    
    return best_model_name, all_results, best_by_type


if __name__ == '__main__':
    best_model, all_results, best_by_type = main()
    
    if best_model:
        print(f"\n최종 결정: {best_model} 모델을 사용합니다.")
    else:
        print("\n모든 모델 테스트에 실패했습니다.")
