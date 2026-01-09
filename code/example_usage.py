# -*- coding: utf-8 -*-
"""
아파트 실거래가 예측 모델 사용 예시
"""
from train_model import ApartmentPricePredictor
from predict import ApartmentPricePredictor as Predictor
import pandas as pd
import numpy as np
import sys
import io

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def example_train():
    """모델 학습 예시"""
    print("=" * 50)
    print("예시 1: 모델 학습")
    print("=" * 50)
    
    # 모델 초기화
    predictor = ApartmentPricePredictor()
    
    # 데이터 로드
    train_df = predictor.load_data('../train.csv')
    
    # 샘플링 (전체 데이터가 너무 클 경우)
    # train_df = train_df.sample(n=10000, random_state=42)
    
    # 데이터 전처리
    X, y = predictor.preprocess_data(train_df, is_train=True)
    
    # 학습/검증 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    predictor.train(
        X_train, y_train,
        X_val, y_val,
        n_estimators=50,  # 빠른 테스트를 위해 작게 설정
        max_depth=15,
        random_state=42
    )
    
    # 모델 저장
    predictor.save_model('apartment_price_model.pkl')
    
    print("\n모델 학습 완료!\n")


def example_predict():
    """예측 예시"""
    print("=" * 50)
    print("예시 2: 예측 실행")
    print("=" * 50)
    
    # 모델 로드
    predictor = Predictor()
    predictor.load_model('apartment_price_model.pkl')
    
    # 테스트 데이터 로드
    try:
        test_df = pd.read_csv('../test.csv', encoding='utf-8')
        print(f"테스트 데이터 shape: {test_df.shape}")
        
        # 예측
        result_df = predictor.predict_from_csv('../test.csv', 'predictions.csv')
        
        print(f"\n예측 완료!")
        print(f"예측된 가격 통계:")
        print(result_df['predicted_price'].describe())
        
    except FileNotFoundError:
        print("test.csv 파일을 찾을 수 없습니다.")
        print("학습 데이터의 일부를 사용하여 예측 예시를 보여드립니다.")
        
        # 학습 데이터의 일부를 사용
        train_df = pd.read_csv('../train.csv', encoding='utf-8')
        sample_df = train_df.drop(columns=['target']).head(10)
        
        # 전처리 및 예측
        X = predictor.preprocess_data(sample_df)
        predictions = predictor.predict(X)
        
        print(f"\n샘플 데이터 예측 결과:")
        for i, pred in enumerate(predictions):
            print(f"  샘플 {i+1}: {pred:,.0f}원")


def example_single_prediction():
    """단일 데이터 예측 예시"""
    print("=" * 50)
    print("예시 3: 단일 데이터 예측")
    print("=" * 50)
    
    # 모델 로드
    predictor = Predictor()
    predictor.load_model('apartment_price_model.pkl')
    
    # 샘플 데이터 생성 (실제 데이터 구조에 맞게 수정 필요)
    sample_data = {
        '시군구': ['서울특별시 강남구 개포동'],
        '번지': ['658-1'],
        '본번': [658.0],
        '부번': [1.0],
        '아파트명': ['개포6차우성'],
        '전용면적(㎡)': [79.97],
        '계약년월': [201712],
        '계약일': [8],
        '층': [3],
        '건축년도': [1987],
        # ... 나머지 컬럼들도 필요에 따라 추가
    }
    
    # 학습 데이터에서 컬럼 구조 확인
    train_df = pd.read_csv('../train.csv', nrows=1, encoding='utf-8')
    sample_df = pd.DataFrame(sample_data)
    
    # 학습 데이터의 모든 컬럼을 포함하도록 (target 제외)
    for col in train_df.columns:
        if col != 'target' and col not in sample_df.columns:
            sample_df[col] = train_df[col].iloc[0] if col in train_df.columns else None
    
    # target 컬럼 제거
    if 'target' in sample_df.columns:
        sample_df = sample_df.drop(columns=['target'])
    
    # 전처리 및 예측
    X = predictor.preprocess_data(sample_df)
    predictions = predictor.predict(X)
    
    print(f"\n예측된 가격: {predictions[0]:,.0f}원")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'train':
            example_train()
        elif mode == 'predict':
            example_predict()
        elif mode == 'single':
            example_single_prediction()
        else:
            print("사용법: python example_usage.py [train|predict|single]")
    else:
        print("사용 가능한 예시:")
        print("1. python example_usage.py train   - 모델 학습")
        print("2. python example_usage.py predict - 예측 실행")
        print("3. python example_usage.py single   - 단일 데이터 예측")
