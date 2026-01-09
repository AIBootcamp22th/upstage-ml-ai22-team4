# -*- coding: utf-8 -*-
"""
아파트 실거래가 예측 모델 학습 스크립트
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
import sys
import io
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os

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
        
    def load_data(self, train_path):
        """데이터 로드"""
        print(f"데이터 로딩 중: {train_path}")
        df = pd.read_csv(train_path, encoding='utf-8')
        print(f"데이터 shape: {df.shape}")
        return df
    
    def preprocess_data(self, df, is_train=True):
        """데이터 전처리"""
        print("데이터 전처리 중...")
        df = df.copy()
        
        # 타겟 변수 분리
        if is_train and self.target_column in df.columns:
            y = df[self.target_column].copy()
            X = df.drop(columns=[self.target_column])
        else:
            y = None
            X = df
        
        # 숫자형 컬럼과 범주형 컬럼 분리
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"숫자형 컬럼: {len(numeric_cols)}개")
        print(f"범주형 컬럼: {len(categorical_cols)}개")
        
        # 숫자형 컬럼 결측치 처리
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # 범주형 컬럼 인코딩
        for col in categorical_cols:
            if is_train:
                # 학습 데이터: LabelEncoder 학습 및 변환
                le = LabelEncoder()
                # 결측치를 'Unknown'으로 처리
                X[col] = X[col].fillna('Unknown').astype(str)
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                # 테스트 데이터: 학습된 LabelEncoder로 변환
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = X[col].fillna('Unknown').astype(str)
                    # 학습 시 보지 못한 값은 'Unknown'으로 처리
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    X[col] = le.transform(X[col])
                else:
                    X[col] = 0
        
        # 모든 컬럼을 숫자형으로 변환
        X = X.select_dtypes(include=[np.number])
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.feature_columns = X.columns.tolist()
        
        if y is not None:
            return X, y
        else:
            return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              n_estimators=100, max_depth=20, random_state=42):
        """모델 학습"""
        print("모델 학습 중...")
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # 학습 데이터 평가
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        print(f"\n=== 학습 데이터 성능 ===")
        print(f"RMSE: {train_rmse:.2f}")
        print(f"MAE: {train_mae:.2f}")
        print(f"R²: {train_r2:.4f}")
        
        # 검증 데이터 평가
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"\n=== 검증 데이터 성능 ===")
            print(f"RMSE: {val_rmse:.2f}")
            print(f"MAE: {val_mae:.2f}")
            print(f"R²: {val_r2:.4f}")
        
        return self.model
    
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 실행하세요.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, model_path='model.pkl'):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 저장되었습니다: {model_path}")
    
    def load_model(self, model_path='model.pkl'):
        """모델 로드"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        print(f"모델이 로드되었습니다: {model_path}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    train_path = '/data/ephemeral/home/py310/train.csv'
    model_save_path = '/data/ephemeral/home/py310/code/apartment_price_model.pkl'
    
    # 모델 초기화
    predictor = ApartmentPricePredictor()
    
    # 데이터 로드
    train_df = predictor.load_data(train_path)
    
    # 데이터 전처리
    X, y = predictor.preprocess_data(train_df, is_train=True)
    
    # 학습/검증 데이터 분할
    print("\n학습/검증 데이터 분할 중...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
    
    # 모델 학습
    predictor.train(
        X_train, y_train, 
        X_val, y_val,
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    
    # 모델 저장
    predictor.save_model(model_save_path)
    
    print("\n모델 학습이 완료되었습니다!")


if __name__ == '__main__':
    main()
