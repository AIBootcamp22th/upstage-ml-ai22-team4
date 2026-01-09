# 아파트 실거래가 예측 모델

이 프로젝트는 `train.csv` 데이터를 사용하여 아파트 실거래가를 예측하는 머신러닝 모델을 생성합니다.

## 파일 구조

- `train_model.py`: 모델 학습 스크립트
- `predict.py`: 예측 스크립트
- `apartment_price_model.pkl`: 학습된 모델 파일 (학습 후 생성됨)

## 사용 방법

### 1. 모델 학습

```bash
cd /data/ephemeral/home/py310/code
python train_model.py
```

이 명령은 `train.csv` 파일을 사용하여 모델을 학습하고 `apartment_price_model.pkl` 파일로 저장합니다.

### 2. 예측 실행

```bash
# 기본 사용법
python predict.py ../test.csv

# 모델 경로와 출력 파일 지정
python predict.py ../test.csv apartment_price_model.pkl predictions.csv
```

### 3. Python 코드에서 직접 사용

```python
from predict import ApartmentPricePredictor
import pandas as pd

# 모델 로드
predictor = ApartmentPricePredictor()
predictor.load_model('apartment_price_model.pkl')

# 데이터 로드
df = pd.read_csv('../test.csv')

# 전처리 및 예측
X = predictor.preprocess_data(df)
predictions = predictor.predict(X)

print(f"예측된 가격: {predictions}")
```

## 모델 정보

- **모델 타입**: Random Forest Regressor
- **입력 데이터**: train.csv
- **타겟 변수**: target (실거래가)
- **평가 지표**: RMSE, MAE, R²

## 주요 기능

1. **자동 데이터 전처리**
   - 숫자형 컬럼 결측치 처리 (중앙값으로 대체)
   - 범주형 컬럼 Label Encoding
   - 무한대 값 처리

2. **모델 학습 및 평가**
   - 학습/검증 데이터 자동 분할
   - 성능 지표 자동 계산

3. **예측 기능**
   - CSV 파일로부터 예측
   - Python 코드에서 직접 사용 가능

## 주의사항

- 모델 학습에는 시간이 걸릴 수 있습니다 (데이터 크기에 따라)
- 메모리 부족 시 배치 처리나 샘플링을 고려하세요
- 학습된 모델과 동일한 전처리 과정을 거쳐야 정확한 예측이 가능합니다
