# -*- coding: utf-8 -*-
"""
아파트 실거래가 예측 스크립트
"""
import pandas as pd
import numpy as np
import pickle
import sys
import os
import io

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ApartmentPricePredictor:
    """아파트 실거래가 예측 모델 클래스"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'target'
        
    def load_model(self, model_path='apartment_price_model.pkl'):
        """모델 로드"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        print(f"모델이 로드되었습니다: {model_path}")
    
    def preprocess_data(self, df):
        """데이터 전처리 (예측용)"""
        print("데이터 전처리 중...")
        X = df.copy()
        
        # 숫자형 컬럼과 범주형 컬럼 분리
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # 숫자형 컬럼 결측치 처리
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # 범주형 컬럼 인코딩
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                X[col] = X[col].fillna('Unknown').astype(str)
                
                # 학습 시 보지 못한 값 처리
                # 'Unknown'이 classes_에 있으면 사용, 없으면 첫 번째 클래스로 매핑
                if len(le.classes_) == 0:
                    X[col] = 0
                else:
                    default_value = le.classes_[0]
                    # classes_를 set으로 변환하여 빠른 조회
                    classes_set = set(le.classes_)
                    
                    # 'Unknown'이 classes_에 있는지 확인
                    if 'Unknown' in classes_set:
                        # 'Unknown'이 있으면, 보지 못한 값은 'Unknown'으로 매핑
                        X[col] = X[col].apply(lambda x: x if x in classes_set else 'Unknown')
                    else:
                        # 'Unknown'이 없으면, 보지 못한 값은 기본값으로 매핑
                        X[col] = X[col].apply(lambda x: x if x in classes_set else default_value)
                    
                    # 변환 수행
                    try:
                        X[col] = le.transform(X[col])
                    except ValueError as e:
                        # 여전히 오류가 발생하면 기본값으로 대체
                        print(f"경고: {col} 컬럼 인코딩 중 오류 발생, 기본값 사용: {e}")
                        X[col] = 0
            else:
                X[col] = 0
        
        # 모든 컬럼을 숫자형으로 변환
        X = X.select_dtypes(include=[np.number])
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 학습 시 사용한 컬럼만 선택 (순서도 맞춤)
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]
        
        return X
    
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model() 메서드를 먼저 실행하세요.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_from_csv(self, csv_path, output_path=None):
        """CSV 파일에서 데이터를 읽어 예측"""
        print(f"데이터 로딩 중: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"데이터 shape: {df.shape}")
        
        # 전처리
        X = self.preprocess_data(df)
        
        # 예측
        print("예측 중...")
        predictions = self.predict(X)
        
        # 결과 저장
        result_df = df.copy()
        result_df['predicted_price'] = predictions
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"예측 결과가 저장되었습니다: {output_path}")
        
        return result_df


def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python predict.py <입력_CSV_파일> [모델_경로] [출력_CSV_파일]")
        print("예시: python predict.py ../test.csv apartment_price_model.pkl predictions.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else '/data/ephemeral/home/py310/code/apartment_price_model.pkl'
    output_csv = sys.argv[3] if len(sys.argv) > 3 else '/data/ephemeral/home/py310/code/predictions.csv'
    
    # 모델 초기화 및 로드
    predictor = ApartmentPricePredictor()
    predictor.load_model(model_path)
    
    # 예측 실행
    result_df = predictor.predict_from_csv(input_csv, output_csv)
    
    print(f"\n예측 완료! 총 {len(result_df)}개 데이터에 대한 예측이 완료되었습니다.")
    print(f"예측 가격 범위: {result_df['predicted_price'].min():.0f} ~ {result_df['predicted_price'].max():.0f}")


if __name__ == '__main__':
    main()
