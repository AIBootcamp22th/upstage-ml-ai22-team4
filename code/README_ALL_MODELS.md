# 모든 모델 최적 파라미터 및 성능 비교

## 테스트 결과 요약

50,000개 데이터 기준 (학습/검증 80/20) 테스트 결과입니다.

### 1. Gradient Boosting (최고 성능)

**최적 파라미터:**
- `n_estimators`: 250
- `learning_rate`: 0.04
- `max_depth`: 8
- `subsample`: 0.85
- `min_samples_split`: 5
- `min_samples_leaf`: 2

**성능:**
- R²: **0.9868**
- RMSE: **4,929.38**
- MAE: 2,807.70
- 학습 시간: ~51초

**특징:**
- 가장 높은 예측 정확도
- 학습 시간이 상대적으로 길지만 가장 우수한 성능

### 2. XGBoost (빠른 속도 + 높은 성능)

**최적 파라미터:**
- `n_estimators`: 250
- `learning_rate`: 0.04
- `max_depth`: 8
- `min_child_weight`: 2
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `tree_method`: 'hist' (메모리 효율성)

**성능:**
- R²: **0.9849**
- RMSE: **5,265.78**
- MAE: 2,893.04
- 학습 시간: **~0.86초**

**특징:**
- 매우 빠른 학습 속도
- Gradient Boosting에 근접한 성능
- 속도 대비 최고의 성능

### 3. Random Forest (균형잡힌 성능)

**최적 파라미터:**
- `n_estimators`: 200
- `max_depth`: 30
- `min_samples_split`: 3
- `min_samples_leaf`: 2
- `max_features`: 'sqrt'

**성능:**
- R²: **0.9837**
- RMSE: **5,472.78**
- MAE: 2,817.48
- 학습 시간: **~1.13초**

**특징:**
- 빠른 학습 속도
- 안정적인 성능
- 기본 모델로 사용하기 좋음

### 4. LightGBM (중간 성능)

**최적 파라미터:**
- `n_estimators`: 250
- `learning_rate`: 0.04
- `num_leaves`: 50
- `min_data_in_leaf`: 15
- `feature_fraction`: 0.8
- `bagging_fraction`: 0.8
- `bagging_freq`: 5

**성능:**
- R²: **0.9829**
- RMSE: **5,602.19**
- MAE: 3,310.38
- 학습 시간: ~22초

**특징:**
- 중간 수준의 성능과 학습 시간
- 범주형 변수 처리에 유리 (현재는 이미 인코딩된 데이터 사용)

## 전체 성능 비교 (상위 10개)

| 모델 | R² | RMSE | MAE | 학습시간(초) |
|------|-------|--------|--------|-------------|
| GB_튜닝1 | 0.9868 | 4,929.38 | 2,807.70 | 51.42 |
| XGB_튜닝1 | 0.9849 | 5,265.78 | 2,893.04 | 0.86 |
| GB_최적1 | 0.9849 | 5,273.15 | 3,128.70 | 34.88 |
| GB_최적2 | 0.9848 | 5,285.49 | 3,147.06 | 52.17 |
| RF_튜닝1 | 0.9837 | 5,472.78 | 2,817.48 | 1.13 |
| XGB_최적1 | 0.9831 | 5,578.54 | 3,200.69 | 0.53 |
| RF_기본 | 0.9830 | 5,593.11 | 2,940.18 | 0.56 |
| RF_최적1 | 0.9829 | 5,600.71 | 2,856.02 | 0.85 |
| LGBM_튜닝1 | 0.9829 | 5,602.19 | 3,310.38 | 21.50 |
| RF_최적2 | 0.9829 | 5,603.38 | 2,855.15 | 1.10 |

## 모델 선택 가이드

### 최고 성능이 필요한 경우
- **Gradient Boosting (GB_튜닝1)**
  - R² 0.9868, RMSE 4,929.38
  - 학습 시간: ~51초
  - 가장 정확한 예측이 필요한 경우

### 속도 대비 최고 성능이 필요한 경우
- **XGBoost (XGB_튜닝1)**
  - R² 0.9849, RMSE 5,265.78
  - 학습 시간: **~0.86초** (매우 빠름)
  - 빠른 학습이 필요하지만 높은 정확도도 원하는 경우

### 효율성 균형이 필요한 경우
- **Random Forest (RF_튜닝1)**
  - R² 0.9837, RMSE 5,472.78
  - 학습 시간: **~1.13초**
  - 기본 모델로 사용하기 좋음

### 중간 성능이 필요한 경우
- **LightGBM (LGBM_튜닝1)**
  - R² 0.9829, RMSE 5,602.19
  - 학습 시간: ~22초
  - 다른 옵션들과 비교했을 때 상대적으로 낮은 성능

## 적용된 전처리 방법

모든 모델에 동일하게 적용된 전처리:
1. **Target Encoding**: 시군구, 아파트명 (R² +0.0096)
2. **Frequency Encoding**: 시군구, 아파트명, 도로명 (R² +0.0039)
3. **연식 및 노후도 파생변수**: 건축년도 기반 (R² +0.0002)
4. **평형대 파생변수**: 전용면적 기반 (5개 카테고리)
5. **역세권 파생변수**: 버스/지하철 역세권 여부 (6개 Boolean)

## 모델별 특별 처리

### LightGBM & XGBoost
- 컬럼명 정리: 특수문자를 언더스코어로 변경하여 호환성 확보
- LightGBM: 범주형 변수 직접 처리 가능 (현재는 이미 인코딩됨)
- XGBoost: `tree_method='hist'`로 메모리 효율성 향상

### Random Forest & Gradient Boosting
- 추가 컬럼명 정리 불필요 (sklearn 기본 호환)
- 원본 데이터 그대로 사용

### 5. 앙상블 모델 (최고 성능)

**현재 시도한 앙상블 방법:**
1. VotingRegressor (평균 앙상블)
2. Weighted Average Ensemble (성능 기반 가중 평균)
3. Simple Average Ensemble (단순 평균)
4. StackingRegressor with RidgeCV
5. StackingRegressor with LassoCV ⭐ (신규)
6. StackingRegressor with ElasticNetCV ⭐ (신규, 최고 성능)
7. StackingRegressor with LinearRegression ⭐ (신규)
8. Blending ⭐ (신규)
9. AdaBoost Regressor ⭐ (신규)
10. ExtraTrees Regressor ⭐ (신규)
11. Bagging Regressor ⭐ (신규)

**신규 추가 앙상블 방법:**

**앙상블 방법:**
1. **StackingRegressor** (메타 모델: ElasticNetCV) ⭐ 최고
   - R²: **0.9873** (최고)
   - RMSE: **4,832.07** (최고)
   - MAE: 2,662.61
   - 학습 시간: ~96초
   
2. **StackingRegressor** (메타 모델: LassoCV)
   - R²: **0.9873**
   - RMSE: **4,834.07**
   - MAE: 2,676.02
   - 학습 시간: ~97초
   
3. **StackingRegressor** (메타 모델: LinearRegression)
   - R²: **0.9873**
   - RMSE: **4,836.87**
   - MAE: 2,679.22
   - 학습 시간: ~96초
   
4. **StackingRegressor** (메타 모델: RidgeCV)
   - R²: **0.9871**
   - RMSE: **4,871.98**
   - MAE: 2,703.42
   - 학습 시간: ~96초
   
2. **VotingRegressor** (평균 앙상블)
   - R²: **0.9870**
   - RMSE: **4,892.46**
   - MAE: 2,638.25
   - 학습 시간: ~53초
   
3. **Weighted Average Ensemble** (성능 기반 가중 평균)
   - R²: **0.9870**
   - RMSE: **4,892.01**
   - MAE: 2,638.23
   - 학습 시간: ~52초
   - 가중치: RF=0.499, GB=0.501
   
4. **Simple Average Ensemble** (단순 평균)
   - R²: **0.9870**
   - RMSE: **4,892.46**
   - MAE: 2,638.25
   - 학습 시간: ~52초
   
5. **Bagging Regressor** ⭐ 신규
   - R²: **0.9851**
   - RMSE: **5,226.75**
   - MAE: 2,681.24
   - 학습 시간: **~3.77초** (빠른 속도 + 좋은 성능)
   
6. **ExtraTrees Regressor** ⭐ 신규
   - R²: **0.9805**
   - RMSE: **5,985.83**
   - MAE: 3,258.46
   - 학습 시간: **~0.65초** (매우 빠름)
   
7. **Blending** ⭐ 신규
   - R²: **0.9789**
   - RMSE: **6,234.07**
   - MAE: 3,009.31
   - 학습 시간: ~37초
   
8. **AdaBoost Regressor** ⭐ 신규
   - R²: **0.9340**
   - RMSE: **11,015.65**
   - MAE: 7,516.34
   - 학습 시간: ~15초

**특징:**
- 모든 앙상블 방법이 개별 모델보다 우수한 성능
- StackingRegressor가 가장 높은 성능 (R² 0.9871)
- VotingRegressor와 Simple Average는 동일한 결과 (가중치가 동일)

## 결론

- **최고 성능**: **앙상블 모델 (StackingRegressor with ElasticNetCV)** ⭐ 최고 추천
  - R² **0.9873**, RMSE **4,832.07**, 학습시간 ~96초
  - Random Forest + Gradient Boosting 조합
  - 메타 모델: ElasticNetCV (5-fold CV)
  - LassoCV, LinearRegression 메타 모델도 동일한 R² 0.9873
  
- **성능과 시간 균형**: **Bagging Regressor** ⭐ 신규 추천
  - R² 0.9851, RMSE 5,226.75, 학습시간 **~3.77초** (매우 빠름)
  - 개별 모델과 근접한 성능, 훨씬 빠른 학습
  
- **균형잡힌 앙상블**: **VotingRegressor / Weighted Average**
  - R² 0.9870, RMSE 4,892, 학습시간 ~52초
  - Stacking보다 빠르고 성능도 우수
  
- **개별 모델 최고**: Gradient Boosting
  - R² 0.9868, RMSE 4,929.38, 학습시간 ~51초
  
- **속도 대비 최고**: XGBoost
  - R² 0.9849, RMSE 5,265.78, 학습시간 ~0.86초
  
- **효율성 균형**: Random Forest
  - R² 0.9837, RMSE 5,472.78, 학습시간 ~1.13초

**주요 발견 사항:**
- StackingRegressor with ElasticNetCV/LassoCV/LinearRegression이 가장 우수 (R² 0.9873)
- Bagging Regressor는 빠른 속도(~3.77초)와 좋은 성능(R² 0.9851)의 균형
- ExtraTrees Regressor는 매우 빠른 학습(~0.65초)으로 실용적
- Blending과 AdaBoost는 상대적으로 낮은 성능 (홀드아웃 세트 크기, 약한 학습기 조합의 한계)

현재 기본 모델은 Random Forest로 설정되어 있으며, 최고 성능을 원하는 경우 앙상블 모델(StackingRegressor with ElasticNetCV), 성능과 시간의 균형을 원하는 경우 Bagging Regressor, 빠른 학습이 필요한 경우 XGBoost 또는 ExtraTrees Regressor를 선택할 수 있습니다.
