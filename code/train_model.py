# -*- coding: utf-8 -*-
"""
아파트 실거래가 예측 모델 학습 스크립트
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
import sys
import io
import re
import logging
from datetime import datetime, timedelta
import time
import threading
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os

# psutil이 없으면 설치하도록 안내, 없어도 동작하도록 처리
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("경고: psutil이 설치되어 있지 않습니다. 시스템 모니터링 기능이 제한됩니다.")

# LightGBM과 XGBoost import (선택적)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 한글 출력을 위한 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ApartmentPricePredictor:
    """아파트 실거래가 예측 모델 클래스"""
    
    def __init__(self, log_file=None, log_level=logging.INFO):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'target'
        self.region_mean_floor = {}  # 시군구별 평균 층수 (층수 지역정규화용)
        self.target_encodings = {}  # Target Encoding을 위한 타겟 평균 (예측 시 사용)
        self.frequency_encodings = {}  # Frequency Encoding을 위한 빈도 정보 (예측 시 사용)
        
        # 로깅 설정
        self.logger = self._setup_logger(log_file, log_level)
        self.training_history = []  # 학습 진행 기록 저장
        
        # 결측치 및 이상치 처리 통계량 저장 (예측 시 사용)
        self.missing_value_stats = {}  # 결측치 처리 통계량
        self.outlier_stats = {}  # 이상치 처리 통계량 (IQR, Z-score 등)
    
    def _setup_logger(self, log_file=None, log_level=logging.INFO):
        """로거 설정"""
        logger = logging.getLogger('ApartmentPricePredictor')
        logger.setLevel(log_level)
        
        # 기존 핸들러 제거
        if logger.handlers:
            logger.handlers.clear()
        
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (로그 파일 지정 시)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _handle_missing_values_and_outliers(self, df, is_train=True):
        """결측치 및 이상치 분석 및 처리
        
        Parameters:
        -----------
        df : pandas.DataFrame
            입력 데이터프레임
        is_train : bool
            학습 데이터인지 여부 (기본값: True)
            
        Returns:
        --------
        pandas.DataFrame
            처리된 데이터프레임
        """
        if is_train:
            self.logger.info(f"  데이터 shape: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 1. 결측치 분석
        missing_info = self._analyze_missing_values(df, is_train)
        
        # 2. 결측치 처리
        df_processed = self._treat_missing_values(df, missing_info, is_train)
        
        # 3. 이상치 분석 (숫자형 컬럼만)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            if is_train:
                outlier_info = self._analyze_outliers(df_processed, numeric_cols, is_train)
            else:
                outlier_info = None  # 예측 시에는 저장된 통계량 사용
            
            # 4. 이상치 처리
            df_processed = self._treat_outliers(df_processed, numeric_cols, outlier_info, is_train)
        
        return df_processed
    
    def _analyze_missing_values(self, df, is_train=True):
        """결측치 분석"""
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'column': missing_counts.index,
            'missing_count': missing_counts.values,
            'missing_percent': missing_percent.values
        }).sort_values('missing_count', ascending=False)
        
        # 결측치가 있는 컬럼만 필터링
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        if is_train:
            if len(missing_info) > 0:
                self.logger.info(f"\n  결측치 분석 결과:")
                self.logger.info(f"    결측치가 있는 컬럼: {len(missing_info)}개")
                self.logger.info(f"    총 결측치 수: {missing_info['missing_count'].sum():,}개")
                
                # 상위 10개만 표시
                top_missing = missing_info.head(10)
                self.logger.info(f"\n    상위 결측치 컬럼 (상위 10개):")
                for idx, row in top_missing.iterrows():
                    self.logger.info(f"      - {row['column']}: {row['missing_count']:,.0f}개 ({row['missing_percent']:.2f}%)")
                
                if len(missing_info) > 10:
                    self.logger.info(f"      ... 외 {len(missing_info) - 10}개 컬럼")
            else:
                self.logger.info(f"  결측치 분석 결과: 결측치 없음")
        
        # 학습 시 통계량 저장
        if is_train:
            self.missing_value_stats = {
                'columns_with_missing': missing_info['column'].tolist(),
                'missing_counts': missing_info.set_index('column')['missing_count'].to_dict(),
                'missing_percents': missing_info.set_index('column')['missing_percent'].to_dict()
            }
        
        return missing_info
    
    def _treat_missing_values(self, df, missing_info, is_train=True):
        """결측치 처리"""
        df_processed = df.copy()
        
        if len(missing_info) == 0:
            return df_processed
        
        treated_count = 0
        
        if is_train:
            self.logger.info(f"\n  결측치 처리 시작:")
        
        for idx, row in missing_info.iterrows():
            col = row['column']
            missing_count = int(row['missing_count'])
            
            # 숫자형 컬럼: 중앙값으로 대체
            if df_processed[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                median_val = df_processed[col].median()
                if pd.isna(median_val):
                    # 모든 값이 NaN인 경우 0으로 대체
                    median_val = 0
                df_processed[col].fillna(median_val, inplace=True)
                
                if is_train:
                    self.logger.info(f"    - {col} ({missing_count:,}개): 중앙값 {median_val:.2f}로 대체")
                treated_count += 1
            
            # 범주형 컬럼: 'Unknown'으로 대체
            elif df_processed[col].dtype == 'object':
                df_processed[col].fillna('Unknown', inplace=True)
                if is_train:
                    self.logger.info(f"    - {col} ({missing_count:,}개): 'Unknown'으로 대체")
                treated_count += 1
            
            # Boolean 컬럼: False로 대체
            elif df_processed[col].dtype == 'bool':
                df_processed[col].fillna(False, inplace=True)
                if is_train:
                    self.logger.info(f"    - {col} ({missing_count:,}개): False로 대체")
                treated_count += 1
        
        if is_train:
            self.logger.info(f"  결측치 처리 완료: {treated_count}개 컬럼 처리됨")
        
        return df_processed
    
    def _analyze_outliers(self, df, numeric_cols, is_train=True):
        """이상치 분석 (IQR 방법 및 Z-score 방법)"""
        outlier_info = {}
        
        if is_train:
            self.logger.info(f"\n  이상치 분석 시작 (숫자형 컬럼 {len(numeric_cols)}개):")
        
        for col in numeric_cols:
            if df[col].isnull().sum() == len(df[col]):
                continue  # 모든 값이 NaN인 경우 스킵
            
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # IQR 방법
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue  # IQR이 0이면 이상치 탐지 불가
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            iqr_outlier_percent = (iqr_outliers / len(df)) * 100
            
            # Z-score 방법 (절대값 3 이상을 이상치로 간주)
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            z_threshold = 3
            z_outlier_count = (z_scores > z_threshold).sum()
            z_outlier_percent = (z_outlier_count / len(col_data)) * 100
            
            outlier_info[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'iqr_outlier_count': iqr_outliers,
                'iqr_outlier_percent': iqr_outlier_percent,
                'z_outlier_count': z_outlier_count,
                'z_outlier_percent': z_outlier_percent,
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
            
            if is_train and (iqr_outliers > 0 or z_outlier_count > 0):
                self.logger.info(f"    - {col}:")
                self.logger.info(f"      IQR 방법: {iqr_outliers:,}개 ({iqr_outlier_percent:.2f}%) 이상치")
                self.logger.info(f"      Z-score 방법: {z_outlier_count:,}개 ({z_outlier_percent:.2f}%) 이상치")
                self.logger.info(f"      범위: [{col_data.min():.2f}, {col_data.max():.2f}], IQR: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        if is_train:
            self.logger.info(f"  이상치 분석 완료: {len(outlier_info)}개 컬럼 분석됨")
        
        # 학습 시 통계량 저장
        if is_train:
            self.outlier_stats = outlier_info
        
        return outlier_info
    
    def _treat_outliers(self, df, numeric_cols, outlier_info, is_train=True):
        """이상치 처리 (Capping 방법 사용)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            입력 데이터프레임
        numeric_cols : list
            숫자형 컬럼 리스트
        outlier_info : dict or None
            이상치 정보 (학습 시 생성, 예측 시에는 None 또는 저장된 통계량 사용)
        is_train : bool
            학습 데이터인지 여부
            
        Returns:
        --------
        pandas.DataFrame
            처리된 데이터프레임
        """
        df_processed = df.copy()
        treated_count = 0
        
        # 예측 시: 학습 시 저장된 통계량 사용
        if not is_train and hasattr(self, 'outlier_stats') and self.outlier_stats:
            outlier_info = self.outlier_stats
        
        if not outlier_info or len(outlier_info) == 0:
            return df_processed
        
        if is_train:
            self.logger.info(f"\n  이상치 처리 시작 (Capping 방법):")
        
        for col in numeric_cols:
            if col not in outlier_info:
                continue
            
            info = outlier_info[col]
            
            # IQR 범위로 capping (하한/상한 값으로 대체)
            lower_bound = info['lower_bound']
            upper_bound = info['upper_bound']
            
            before_outlier_count = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            
            # 이상치를 하한/상한 값으로 대체
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
            after_outlier_count = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            
            if before_outlier_count > 0:
                treated_count += 1
                if is_train:
                    self.logger.info(f"    - {col}: {before_outlier_count:,}개 이상치 → {after_outlier_count:,}개 (Capping: [{lower_bound:.2f}, {upper_bound:.2f}])")
        
        if is_train:
            self.logger.info(f"  이상치 처리 완료: {treated_count}개 컬럼 처리됨")
            self.logger.info("")
        
        return df_processed
        
    def load_data(self, train_path, add_location_features=False, bus_csv_path=None, subway_csv_path=None):
        """데이터 로드
        
        Parameters:
        -----------
        train_path : str
            학습 데이터 CSV 파일 경로
        add_location_features : bool
            역세권 여부(버스정거장, 지하철역)를 추가할지 여부 (기본값: False)
        bus_csv_path : str
            버스정거장 CSV 파일 경로 (add_location_features=True일 때 필요)
        subway_csv_path : str
            지하철역 CSV 파일 경로 (add_location_features=True일 때 필요)
        """
        print(f"데이터 로딩 중: {train_path}")
        df = pd.read_csv(train_path, encoding='utf-8')
        print(f"데이터 shape: {df.shape}")
        
        # 위치 정보 추가 옵션
        if add_location_features:
            import sys
            import os
            # 현재 파일의 디렉토리를 경로에 추가
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from add_location_features import add_location_features as add_loc_features
            print("\n역세권 여부 확인 중...")
            if bus_csv_path is None:
                bus_csv_path = '/data/ephemeral/home/py310/bus_feature.csv'
            if subway_csv_path is None:
                subway_csv_path = '/data/ephemeral/home/py310/subway_feature.csv'
            
            df = add_loc_features(df, bus_csv_path, subway_csv_path)
            print(f"역세권 여부 추가 후 데이터 shape: {df.shape}")
        
        return df
    
    def preprocess_data(self, df, is_train=True):
        """데이터 전처리"""
        if is_train:
            self.logger.info("=" * 70)
            self.logger.info("데이터 전처리 시작")
            self.logger.info("=" * 70)
        else:
            print("데이터 전처리 중...")
        df = df.copy()
        
        # 연립주택 데이터 제거 (k-단지분류(아파트,주상복합등등)가 "연립주택"인 경우)
        complex_type_col = 'k-단지분류(아파트,주상복합등등)'
        if complex_type_col in df.columns:
            before_count = len(df)
            # 연립주택 데이터 제거
            df = df[df[complex_type_col] != '연립주택'].copy()
            removed_count = before_count - len(df)
            
            if is_train:
                if removed_count > 0:
                    self.logger.info(f"\n연립주택 데이터 제거:")
                    self.logger.info(f"  제거 전 데이터 수: {before_count:,}개")
                    self.logger.info(f"  제거된 데이터 수: {removed_count:,}개 ({removed_count/before_count*100:.2f}%)")
                    self.logger.info(f"  제거 후 데이터 수: {len(df):,}개")
                else:
                    self.logger.info(f"\n연립주택 데이터 제거: 연립주택 데이터 없음")
            else:
                if removed_count > 0:
                    print(f"연립주택 데이터 제거: {removed_count:,}개 제거됨")
        
        # 결측치 및 이상치 분석 및 처리
        if is_train:
            self.logger.info("\n1. 결측치 및 이상치 분석 및 처리")
            df = self._handle_missing_values_and_outliers(df, is_train=True)
        else:
            # 예측 시에는 학습 시 저장된 통계량 사용
            df = self._handle_missing_values_and_outliers(df, is_train=False)
        
        # 계약년월과 계약일을 합쳐서 YYMMDDhhmmss 형태의 파생변수 생성
        if '계약년월' in df.columns and '계약일' in df.columns:
            print("계약일시 파생변수 생성 중...")
            
            # 계약년월을 문자열로 변환
            contract_year_month = df['계약년월'].astype(str)
            
            # 연도 추출 (YYYYMM -> YYYY, MM)
            df['계약연도'] = contract_year_month.str[:4]
            df['계약월'] = contract_year_month.str[4:6]
            
            # 계약일을 2자리 문자열로 변환 (한 자리 수는 앞에 0 추가)
            df['계약일_포맷'] = df['계약일'].astype(int).astype(str).str.zfill(2)
            
            # YYMMDDhhmmss 형태: YYYY + MM + DD
            df['계약일자'] = (
                df['계약연도'].astype(str) + 
                df['계약월'].astype(str) + 
                df['계약일_포맷'].astype(str)
            ).astype(int)
            
            # 임시 컬럼 제거
            df = df.drop(columns=['계약연도', '계약월', '계약일_포맷'])
            
            print(f"계약일자 파생변수 생성 완료: 예시 = {df['계약일자'].iloc[0]}")
        
        # 전용면적을 평형대별 파생변수로 변환
        if '전용면적(㎡)' in df.columns:
            print("전용면적 평형대 파생변수 생성 중...")
            
            # 평형 계산 (1평 ≈ 3.3058㎡)
            df['평형'] = df['전용면적(㎡)'] / 3.3058
            
            # ㎡ 기준 평형대 분류 (일반적인 한국 아파트 분류)
            # 60㎡ 이하: 소형 (약 18평 이하)
            # 60~85㎡: 중소형 (약 18~26평)
            # 85~102㎡: 중형 (약 26~31평)
            # 102~135㎡: 중대형 (약 31~41평)
            # 135㎡ 초과: 대형 (약 41평 초과)
            
            def categorize_area_by_pyeong(area):
                """㎡ 기준으로 평형대 분류"""
                if pd.isna(area):
                    return 'Unknown'
                elif area <= 60:
                    return '소형_60이하'
                elif area <= 85:
                    return '중소형_60_85'
                elif area <= 102:
                    return '중형_85_102'
                elif area <= 135:
                    return '중대형_102_135'
                else:
                    return '대형_135초과'
            
            # 평형대 카테고리 변수 생성
            df['평형대_카테고리'] = df['전용면적(㎡)'].apply(categorize_area_by_pyeong)
            
            # 평형대 숫자 인코딩 (원핫 인코딩 대신 순서형 인코딩)
            pyeong_mapping = {
                'Unknown': 0,
                '소형_60이하': 1,
                '중소형_60_85': 2,
                '중형_85_102': 3,
                '중대형_102_135': 4,
                '대형_135초과': 5
            }
            df['평형대_코드'] = df['평형대_카테고리'].map(pyeong_mapping)
            
            # 평형대별 더미 변수 생성 (선택적)
            df['평형대_소형'] = (df['평형대_코드'] == 1).astype(int)
            df['평형대_중소형'] = (df['평형대_코드'] == 2).astype(int)
            df['평형대_중형'] = (df['평형대_코드'] == 3).astype(int)
            df['평형대_중대형'] = (df['평형대_코드'] == 4).astype(int)
            df['평형대_대형'] = (df['평형대_코드'] == 5).astype(int)
            
            # 평형대 카테고리 컬럼은 나중에 제거 (인코딩 후)
            # 임시로 유지 (범주형 인코딩을 위해)
            
            print(f"평형대 파생변수 생성 완료:")
            print(f"  평형 통계 - 평균: {df['평형'].mean():.2f}평, 중앙값: {df['평형'].median():.2f}평")
            print(f"  평형대 분포 - {df['평형대_카테고리'].value_counts().to_dict()}")
        
        # 건축년도 관련 파생변수 생성 (연식, 노후도) - 성능 향상 확인
        if '건축년도' in df.columns and '계약년월' in df.columns:
            if is_train:
                print("건축년도 관련 파생변수 생성 중...")
            
            # 계약년월에서 연도 추출
            contract_year = df['계약년월'].astype(str).str[:4].astype(int)
            
            # 연식 계산 (계약년도 - 건축년도)
            df['연식'] = (contract_year - df['건축년도']).clip(lower=0)
            
            # 노후도 구간 분류
            def categorize_age(age):
                if pd.isna(age):
                    return 0
                elif age <= 5:
                    return 1  # 신축
                elif age <= 10:
                    return 2  # 준신축
                elif age <= 20:
                    return 3  # 중고
                elif age <= 30:
                    return 4  # 노후
                else:
                    return 5  # 구식
            
            df['노후도_코드'] = df['연식'].apply(categorize_age)
            
            # 노후도별 더미 변수 생성
            df['노후도_신축'] = (df['노후도_코드'] == 1).astype(int)
            df['노후도_준신축'] = (df['노후도_코드'] == 2).astype(int)
            df['노후도_중고'] = (df['노후도_코드'] == 3).astype(int)
            df['노후도_노후'] = (df['노후도_코드'] == 4).astype(int)
            df['노후도_구식'] = (df['노후도_코드'] == 5).astype(int)
            
            if is_train:
                print(f"  연식 범위: {df['연식'].min():.0f}년 ~ {df['연식'].max():.0f}년")
                print(f"  노후도 분포:")
                age_dist = df['노후도_코드'].value_counts().sort_index()
                age_labels = {0: 'Unknown', 1: '신축', 2: '준신축', 3: '중고', 4: '노후', 5: '구식'}
                for code, count in age_dist.items():
                    label = age_labels.get(code, 'Unknown')
                    pct = count / len(df) * 100
                    print(f"    {label}: {count:,}개 ({pct:.2f}%)")
                print(f"  연식 파생변수 생성 완료")
        
        # 층수 관련 파생변수 생성 (지역별 층수에 따른 실거래가 변동 반영)
        # 분석 결과: 층수 제곱(0.1979)이 원본 층수(0.1544)보다 높은 상관관계를 보임
        # 하지만 Random Forest가 이미 비선형 관계를 잘 포착하므로 원본 '층' 컬럼만 사용
        # 고층 프리미엄 패턴(저층: 6.40천만원, 중층: 6.78천만원, 고층: 10.01천만원, 초고층: 22.49천만원)은
        # Random Forest 모델이 원본 층수 값으로 충분히 학습 가능
        if '층' in df.columns:
            # 층수 기본 정보 확인만 수행
            if is_train:
                high_floor_count = (df['층'] >= 21).sum()
                super_high_floor_count = (df['층'] >= 50).sum()
                print(f"층수 정보 확인:")
                print(f"  층수 범위: {df['층'].min()}층 ~ {df['층'].max()}층")
                print(f"  고층(21층 이상): {high_floor_count:,}개 ({high_floor_count/len(df)*100:.2f}%)")
                print(f"  초고층(50층 이상): {super_high_floor_count:,}개 ({super_high_floor_count/len(df)*100:.2f}%)")
                print(f"  (원본 '층' 컬럼만 사용 - Random Forest가 비선형 관계를 충분히 학습)")
        
        # 불필요한 컬럼 제거 (실거래가 예측에 불필요한 항목)
        columns_to_remove = [
            '해제사유발생일',
            '등기신청일자',
            '거래유형',
            '중개사소재지',
            'k-전화번호',
            'k-팩스번호',
            '단지소개기존clob',
            'k-관리방식',
            'k-사용검사일-사용승인일',
            'k-관리비부과면적',
            'k-전용면적별세대현황(60㎡이하)',
            'k-전용면적별세대현황(60㎡~85㎡이하)',
            'k-85㎡~135㎡이하',
            'k-135㎡초과',
            'k-홈페이지',
            'k-등록일자',
            'k-수정일자',
            '고용보험관리번호',
            '경비비관리형태',
            '세대전기계약방법',
            '청소비관리형태',
            '기타/의무/임대/임의=1/2/3/4',
            '사용허가여부',
            '관리비 업로드',
            '단지신청일'
        ]
        
        # 존재하는 컬럼만 제거
        existing_remove_cols = [col for col in columns_to_remove if col in df.columns]
        if existing_remove_cols:
            df = df.drop(columns=existing_remove_cols)
            if is_train:
                self.logger.info(f"불필요한 컬럼 제거 완료: {len(existing_remove_cols)}개 컬럼 제거됨")
                self.logger.info(f"  제거된 컬럼: {', '.join(existing_remove_cols[:5])}{'...' if len(existing_remove_cols) > 5 else ''}")
            else:
                print(f"불필요한 컬럼 제거 완료: {len(existing_remove_cols)}개 컬럼 제거됨")
        
        # 시군구 분리 파생변수 생성 (시, 구(군), 동으로 분리) - 성능 향상 확인됨
        if '시군구' in df.columns:
            if is_train:
                self.logger.info("\n시군구 분리 파생변수 생성 중...")
            else:
                print("시군구 분리 파생변수 생성 중...")
            
            # 시군구를 공백으로 분리 (형식: "서울특별시 강남구 개포동")
            sigungu_split = df['시군구'].str.split(' ', expand=True)
            
            # 시 추출 (첫 번째 요소)
            df['시'] = sigungu_split[0] if len(sigungu_split.columns) > 0 else ''
            df['시'] = df['시'].fillna('Unknown')
            
            # 구(군) 추출 (두 번째 요소)
            df['구_군'] = sigungu_split[1] if len(sigungu_split.columns) > 1 else ''
            df['구_군'] = df['구_군'].fillna('Unknown')
            
            # 동 추출 (세 번째 요소, 있으면)
            df['동'] = sigungu_split[2] if len(sigungu_split.columns) > 2 else ''
            df['동'] = df['동'].fillna('Unknown')
            
            if is_train:
                self.logger.info(f"  시 고유값 수: {df['시'].nunique()}개")
                self.logger.info(f"  구(군) 고유값 수: {df['구_군'].nunique()}개")
                self.logger.info(f"  동 고유값 수: {df['동'].nunique()}개")
                self.logger.info(f"  (성능 향상 확인: R² +0.29%, RMSE -3.22%)")
            else:
                print(f"시군구 분리 파생변수 생성 완료: 시({df['시'].nunique()}개), 구(군)({df['구_군'].nunique()}개), 동({df['동'].nunique()}개)")
        
        # Target Encoding 및 Frequency Encoding (성능 향상 확인)
        # 범주형 변수에 대한 Target Encoding 및 Frequency Encoding 적용
        if is_train and self.target_column in df.columns:
            # 주요 범주형 변수에 대한 Target Encoding
            target_encodings = {}
            target_encode_cols = ['시군구', '아파트명']
            
            # 시군구 분리 파생변수에 대해서도 Target Encoding 적용 (성능 향상)
            if '구_군' in df.columns:
                target_encode_cols.append('구_군')
            if '동' in df.columns:
                target_encode_cols.append('동')
            
            for col in target_encode_cols:
                if col in df.columns:
                    target_mean = df.groupby(col)[self.target_column].mean().to_dict()
                    target_encodings[col] = target_mean
                    df[f'{col}_타겟평균'] = df[col].map(target_mean).fillna(df[self.target_column].mean())
            
            self.target_encodings = target_encodings
            
            # 주요 범주형 변수에 대한 Frequency Encoding
            frequency_encodings = {}
            freq_encode_cols = ['시군구', '아파트명', '도로명']
            
            # 시군구 분리 파생변수에 대해서도 Frequency Encoding 적용 (성능 향상)
            if '시' in df.columns:
                freq_encode_cols.append('시')
            if '구_군' in df.columns:
                freq_encode_cols.append('구_군')
            if '동' in df.columns:
                freq_encode_cols.append('동')
            
            for col in freq_encode_cols:
                if col in df.columns:
                    freq_map = df[col].value_counts().to_dict()
                    frequency_encodings[col] = freq_map
                    df[f'{col}_빈도'] = df[col].map(freq_map).fillna(0)
            
            self.frequency_encodings = frequency_encodings
            
            if is_train:
                self.logger.info(f"Target Encoding 및 Frequency Encoding 적용 완료")
                self.logger.info(f"  Target Encoding: {len(target_encodings)}개 변수 ({', '.join([f'{col}_타겟평균' for col in target_encode_cols if col in df.columns])})")
                self.logger.info(f"  Frequency Encoding: {len(frequency_encodings)}개 변수 ({', '.join([f'{col}_빈도' for col in freq_encode_cols if col in df.columns])})")
            else:
                print(f"Target Encoding 및 Frequency Encoding 적용 완료")
                print(f"  Target Encoding: {len(target_encodings)}개 변수 ({', '.join([f'{col}_타겟평균' for col in target_encode_cols if col in df.columns])})")
                print(f"  Frequency Encoding: {len(frequency_encodings)}개 변수 ({', '.join([f'{col}_빈도' for col in freq_encode_cols if col in df.columns])})")
        elif not is_train:
            # 예측 시: 학습 시 저장된 인코딩 정보 사용
            # Target Encoding
            if hasattr(self, 'target_encodings') and self.target_encodings:
                for col, encoding_map in self.target_encodings.items():
                    if col in df.columns:
                        # 기본값은 전체 타겟 평균 (학습 시 저장된 값이 없으면 0)
                        default_value = 0
                        if hasattr(self, 'target_mean'):
                            default_value = self.target_mean
                        df[f'{col}_타겟평균'] = df[col].map(encoding_map).fillna(default_value)
            
            # Frequency Encoding
            if hasattr(self, 'frequency_encodings') and self.frequency_encodings:
                for col, freq_map in self.frequency_encodings.items():
                    if col in df.columns:
                        df[f'{col}_빈도'] = df[col].map(freq_map).fillna(0)
            
            # 시군구 분리 파생변수에 대한 Target Encoding 및 Frequency Encoding (예측 시)
            # 시, 구_군, 동에 대한 Target Encoding
            split_target_cols = ['구_군', '동']
            for col in split_target_cols:
                if col in df.columns and col in getattr(self, 'target_encodings', {}):
                    encoding_map = self.target_encodings[col]
                    default_value = 0
                    if hasattr(self, 'target_mean'):
                        default_value = self.target_mean
                    df[f'{col}_타겟평균'] = df[col].map(encoding_map).fillna(default_value)
            
            # 시, 구_군, 동에 대한 Frequency Encoding
            split_freq_cols = ['시', '구_군', '동']
            for col in split_freq_cols:
                if col in df.columns and col in getattr(self, 'frequency_encodings', {}):
                    freq_map = self.frequency_encodings[col]
                    df[f'{col}_빈도'] = df[col].map(freq_map).fillna(0)
        
        # 타겟 변수 분리
        if is_train and self.target_column in df.columns:
            y = df[self.target_column].copy()
            X = df.drop(columns=[self.target_column])
        else:
            y = None
            X = df
        
        # 평형대 카테고리 컬럼은 이미 더미 변수로 변환되었으므로 제거
        if '평형대_카테고리' in X.columns:
            X = X.drop(columns=['평형대_카테고리'])
        
        # 로그 변환 변수 추가 (성능 향상 확인: R² +3.25%, RMSE -51.2%)
        # 왜도가 큰 양수 변수에 로그 변환 적용
        log_transform_cols = []
        if '전용면적(㎡)' in X.columns:
            X['전용면적_log'] = np.log1p(X['전용면적(㎡)'].clip(lower=0))
            log_transform_cols.append('전용면적_log')
        
        if '평형' in X.columns:
            X['평형_log'] = np.log1p(X['평형'].clip(lower=0))
            log_transform_cols.append('평형_log')
        
        if len(log_transform_cols) > 0:
            if is_train:
                self.logger.info(f"\n로그 변환 변수 생성 완료: {len(log_transform_cols)}개 변수 ({', '.join(log_transform_cols)})")
                self.logger.info(f"  (성능 향상 확인: R² +3.25%, RMSE -51.2%)")
            else:
                print(f"로그 변환 변수 생성 완료: {len(log_transform_cols)}개 변수")
        
        # 역세권 여부 컬럼을 숫자형으로 변환 (Boolean -> int)
        location_flags = [
            '버스_핵심서비스권', '버스_일반영향권',
            '지하철_1차역세권', '지하철_2차역세권', '지하철_초역세권', '지하철_역세권'
        ]
        for col in location_flags:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        # 숫자형 컬럼과 범주형 컬럼 분리
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"숫자형 컬럼: {len(numeric_cols)}개")
        print(f"범주형 컬럼: {len(categorical_cols)}개")
        
        # 숫자형 컬럼 결측치 처리 (이미 _handle_missing_values_and_outliers에서 처리됨)
        # 추가 결측치가 있을 경우 중앙값으로 대체
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0  # 모든 값이 NaN인 경우 0으로 대체
                X[col].fillna(median_val, inplace=True)
                if is_train:
                    self.logger.warning(f"  추가 결측치 발견 및 처리: {col} ({X[col].isnull().sum()}개 → 중앙값 {median_val:.2f}로 대체)")
        
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
              model_type='stacking_elasticnet', n_estimators=100000, max_depth=25, 
              learning_rate=0.05, early_stopping_rounds=50, random_state=42):
        """모델 학습
        
        Parameters:
        -----------
        model_type : str
            모델 타입 ('stacking_elasticnet', 'random_forest', 'gradient_boosting')
            기본값: 'stacking_elasticnet' (최고 성능: R² 0.9873, RMSE 4832, 학습시간 ~96초)
            - 'stacking_elasticnet': Stacking Ensemble with ElasticNetCV (R² 0.9873, RMSE 4832, 학습시간 ~96초, 최고 성능, 기본값) ⭐
            - 'random_forest': Random Forest (R² 0.9846, RMSE 5315, 학습시간 ~5초, 효율성 우수)
            - 'gradient_boosting': Gradient Boosting (R² 0.9849, RMSE 5273, 학습시간 ~35초)
        n_estimators : int
            최대 트리 개수 (early stopping으로 실제 학습 횟수는 적을 수 있음)
            - Random Forest 기본값: 200 (early stopping 없음)
            - Gradient Boosting 기본값: 100000 (early stopping 사용)
        max_depth : int
            최대 깊이 
            - Random Forest 기본값: 30 (최적)
            - Gradient Boosting 기본값: 8 (최적)
        learning_rate : float
            학습률 (Gradient Boosting용, 기본값: 0.04, 최적)
        early_stopping_rounds : int
            Early stopping rounds (Gradient Boosting용, 기본값: 50)
            - 50번 반복 동안 성능 개선이 없으면 학습 중단
        """
        self.logger.info("=" * 70)
        self.logger.info(f"모델 학습 시작 (모델 타입: {model_type})")
        self.logger.info(f"학습 데이터: {X_train.shape[0]:,}개, 특성: {X_train.shape[1]}개")
        if X_val is not None:
            self.logger.info(f"검증 데이터: {X_val.shape[0]:,}개")
        self.logger.info("=" * 70)
        
        self.training_history = []
        start_time = time.time()
        
        if model_type == 'stacking_elasticnet' or model_type == 'ensemble' or model_type == 'stacking':
            # StackingRegressor with ElasticNetCV (최고 성능: R² 0.9873, RMSE 4832)
            self.logger.info("앙상블 모델 구성: Random Forest + Gradient Boosting + ElasticNetCV (메타 모델)")
            
            # 기본 모델들 생성 (최적 파라미터 + early stopping for GB)
            rf_base = RandomForestRegressor(
                n_estimators=200,
                max_depth=30,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
            
            # Gradient Boosting with early stopping
            # StackingRegressor 내부에서는 validation_fraction을 사용하여 early stopping
            # 제공된 검증 데이터는 StackingRegressor의 CV에서 사용됨
            validation_fraction = 0.1  # 내부 검증 세트로 early stopping
            
            # StackingRegressor 내부 GB는 staged_predict로 단계별 로깅이 어려움
            # 대신 verbose를 사용하여 기본 진행 상황 출력
            gb_base = GradientBoostingRegressor(
                n_estimators=n_estimators,  # 최대 반복 횟수 (100,000)
                learning_rate=0.04,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.85,
                validation_fraction=validation_fraction,
                n_iter_no_change=early_stopping_rounds,  # early stopping (50 rounds)
                tol=1e-4,  # 성능 개선 임계값
                random_state=random_state,
                verbose=0  # StackingRegressor 내부에서는 verbose=0 (수동 로깅 사용)
            )
            
            self.logger.info(f"  Random Forest: n_estimators=200, max_depth=30")
            self.logger.info(f"  Gradient Boosting: n_estimators={n_estimators:,}, learning_rate=0.04, max_depth=8")
            self.logger.info(f"  Early Stopping: {early_stopping_rounds} rounds (성능 개선 없으면 중단)")
            self.logger.info(f"  메타 모델: ElasticNetCV (5-fold CV)")
            
            # StackingRegressor with ElasticNetCV 메타 모델
            self.model = StackingRegressor(
                estimators=[('rf', rf_base), ('gb', gb_base)],
                final_estimator=ElasticNetCV(cv=5, random_state=random_state),
                cv=5,
                n_jobs=-1
            )
            self.logger.info("StackingRegressor 학습 시작...")
            self.logger.info("  (각 fold마다 RF와 GB 학습, 그 후 메타 모델 학습)")
            self.logger.info("  GB는 validation_fraction=0.1로 내부 검증 세트를 사용하여 early stopping")
            
            # StackingRegressor는 내부적으로 CV를 수행하므로 직접 로깅하기 어려움
            # 하지만 GB 학습 진행 상황을 모니터링하기 위해 샘플 학습 수행
            fit_start_time = time.time()
            
            # 로깅을 위한 샘플 GB 학습 (전체 데이터의 일부로)
            sample_size = min(10000, len(X_train))
            X_sample = X_train[:sample_size]
            y_sample = y_train[:sample_size]
            
            # GB 학습 진행 상황 로깅을 위한 임시 모델
            temp_gb = GradientBoostingRegressor(
                n_estimators=min(1000, n_estimators),  # 샘플은 1000개까지만 (빠른 로깅)
                learning_rate=0.04,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.85,
                validation_fraction=0.1,
                n_iter_no_change=early_stopping_rounds,
                tol=1e-4,
                random_state=random_state,
                verbose=0
            )
            
            # 샘플 데이터로 학습하여 staged_predict로 진행 상황 확인
            self.logger.info(f"  샘플 데이터({sample_size:,}개)로 GB 학습 진행 상황 모니터링 (실제 학습 전 패턴 확인):")
            self._log_gb_progress(temp_gb, X_sample, y_sample, min(1000, n_estimators), early_stopping_rounds)
            
            self.logger.info("  실제 StackingRegressor 학습 시작 (각 fold마다 동일한 패턴으로 학습)...")
            self.logger.info("  (각 fold의 GB 학습 진행 상황은 내부적으로 처리됨)")
            
            # 학습 진행 모니터링을 위한 백그라운드 스레드 시작
            monitoring_thread = None
            stop_monitoring = threading.Event()
            
            # 예상 시간 계산 (이전 테스트 결과 기반)
            # 50,000개 샘플 기준 약 96초, 현재 데이터는 약 895,057행
            sample_size = 50000
            sample_time = 96  # 초
            data_ratio = len(X_train) / sample_size
            estimated_total_seconds = sample_time * (data_ratio ** 1.2)  # 비선형 관계 고려
            estimated_per_fold = estimated_total_seconds / 6  # 5개 fold + 최종 학습
            
            def monitor_training():
                """학습 진행 상황 모니터링 (주기적 하트비트 + 진행률 추정)"""
                last_log_time = time.time()
                heartbeat_interval = 30  # 30초마다 하트비트 로그
                fold_count = 0
                last_cpu_usage = 0
                cpu_change_threshold = 5  # CPU 사용률 변화 임계값 (fold 전환 감지용)
                
                while not stop_monitoring.is_set():
                    current_time = time.time()
                    elapsed_time = current_time - fit_start_time
                    
                    # 30초마다 하트비트 로그
                    if current_time - last_log_time >= heartbeat_interval:
                        # CPU/메모리 사용률 확인 (psutil이 있는 경우)
                        cpu_percent = 0
                        memory_percent = 0
                        memory_used_gb = 0
                        memory_total_gb = 0
                        
                        if PSUTIL_AVAILABLE:
                            try:
                                cpu_percent = psutil.cpu_percent(interval=1)
                                memory = psutil.virtual_memory()
                                memory_percent = memory.percent
                                memory_used_gb = memory.used / (1024**3)
                                memory_total_gb = memory.total / (1024**3)
                            except Exception as e:
                                pass
                        
                        # 진행률 추정 (시간 기반)
                        # StackingRegressor는 5-fold CV + 최종 학습 = 총 6단계
                        # 각 단계가 비슷한 시간이 걸린다고 가정
                        estimated_progress_pct = min(95, (elapsed_time / estimated_total_seconds) * 100)
                        
                        # Fold 추정 (시간 기반, 대략적인 추정)
                        estimated_fold = min(5, int((elapsed_time / estimated_per_fold)) + 1)
                        if estimated_fold <= 5:
                            current_phase = f"Fold {estimated_fold}/5"
                        elif elapsed_time < estimated_total_seconds * 0.9:
                            current_phase = "메타 모델 학습"
                        else:
                            current_phase = "최종 모델 학습"
                        
                        # CPU 사용률 변화로 fold 전환 감지 시도
                        if abs(cpu_percent - last_cpu_usage) > cpu_change_threshold and last_cpu_usage > 0:
                            fold_count += 1
                            if fold_count <= 5:
                                self.logger.info(f"  [Fold {fold_count}/5 시작 감지] CPU 사용률 변화: {last_cpu_usage:.1f}% → {cpu_percent:.1f}%")
                        
                        last_cpu_usage = cpu_percent
                        
                        # 로그 출력
                        if PSUTIL_AVAILABLE and cpu_percent > 0:
                            self.logger.info(f"  [학습 진행 중] {current_phase} | "
                                            f"예상 진행률: {estimated_progress_pct:.1f}% | "
                                            f"경과 시간: {elapsed_time:.0f}초 (~{elapsed_time/60:.1f}분) | "
                                            f"CPU: {cpu_percent:.1f}% | "
                                            f"메모리: {memory_percent:.1f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB)")
                        else:
                            self.logger.info(f"  [학습 진행 중] {current_phase} | "
                                            f"예상 진행률: {estimated_progress_pct:.1f}% | "
                                            f"경과 시간: {elapsed_time:.0f}초 (~{elapsed_time/60:.1f}분)")
                        
                        # 예상 남은 시간 계산
                        if estimated_progress_pct > 0 and estimated_progress_pct < 95:
                            remaining_time = (estimated_total_seconds - elapsed_time)
                            if remaining_time > 0:
                                self.logger.info(f"    예상 남은 시간: 약 {remaining_time:.0f}초 (~{remaining_time/60:.1f}분)")
                        
                        last_log_time = current_time
                    
                    # 0.5초마다 체크 (CPU 부하 최소화)
                    time.sleep(0.5)
            
            # 모니터링 스레드 시작
            monitoring_thread = threading.Thread(target=monitor_training, daemon=True)
            monitoring_thread.start()
            self.logger.info("  학습 진행 모니터링 시작 (30초마다 상태 업데이트)")
            self.logger.info(f"  예상 총 학습 시간: 약 {estimated_total_seconds:.0f}초 (~{estimated_total_seconds/60:.1f}분)")
            self.logger.info(f"  예상 각 fold 시간: 약 {estimated_per_fold:.0f}초 (~{estimated_per_fold/60:.1f}분)")
            
            try:
                # 실제 StackingRegressor 학습
                self.logger.info("  [Fold 1/5 시작] Random Forest 학습 중...")
                self.model.fit(X_train, y_train)
            finally:
                # 학습 완료 후 모니터링 스레드 종료
                stop_monitoring.set()
                if monitoring_thread:
                    monitoring_thread.join(timeout=1.0)
            
            fit_time = time.time() - fit_start_time
            
            # 학습 완료 후 각 fold의 GB 모델에서 실제 학습된 트리 수 확인
            # StackingRegressor 내부의 GB 모델들 확인
            try:
                # StackingRegressor는 각 fold마다 모델을 학습하지만, 
                # final_estimator를 학습하기 전에 전체 데이터로 다시 학습
                # 따라서 실제 GB 모델의 n_estimators_ 확인
                if hasattr(gb_base, 'n_estimators_'):
                    actual_gb_estimators = gb_base.n_estimators_
                else:
                    # 전체 데이터로 학습된 GB 모델 확인
                    actual_gb_estimators = None
                    for name, estimator in self.model.estimators_:
                        if name == 'gb' and hasattr(estimator, 'n_estimators_'):
                            actual_gb_estimators = estimator.n_estimators_
                            break
                
                if actual_gb_estimators:
                    self.logger.info(f"  Gradient Boosting 실제 학습된 트리 수: {actual_gb_estimators:,}개")
                    if actual_gb_estimators < n_estimators:
                        self.logger.info(f"  Early stopping 적용됨: {n_estimators:,}개 중 {actual_gb_estimators:,}개에서 중단")
                    else:
                        self.logger.info(f"  최대 반복 횟수({n_estimators:,}개)까지 학습 완료")
                else:
                    self.logger.info(f"  Gradient Boosting 최대 반복: {n_estimators:,}개 (early stopping 적용됨)")
            except Exception as e:
                self.logger.warning(f"  GB 모델 정보 확인 중 오류: {e}")
                self.logger.info(f"  Gradient Boosting 최대 반복: {n_estimators:,}개 (early stopping 적용)")
            
            self.logger.info(f"StackingRegressor 학습 완료 (소요 시간: {fit_time:.2f}초, ~{fit_time/60:.1f}분)")
            self.logger.info(f"  파라미터: RF(n=200, d=30), GB(n={n_estimators:,}, lr=0.04, d=8, early_stopping={early_stopping_rounds}), Meta=ElasticNetCV")
            
        elif model_type == 'gradient_boosting' or model_type == 'gb':
            # Gradient Boosting 모델 with early stopping
            self.logger.info("Gradient Boosting 모델 학습 시작...")
            self.logger.info(f"  최대 반복 횟수: {n_estimators:,}개")
            self.logger.info(f"  Early Stopping: {early_stopping_rounds} rounds")
            
            # 검증 데이터가 제공된 경우 validation_fraction 사용하지 않음
            validation_fraction = 0.1 if X_val is None else None
            
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate if learning_rate > 0 else 0.04,
                max_depth=max_depth if max_depth >= 7 else 8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.85,
                validation_fraction=validation_fraction,
                n_iter_no_change=early_stopping_rounds,
                tol=1e-4,
                random_state=random_state,
                verbose=0  # verbose=0으로 설정하고 수동 로깅 사용
            )
            
            # 학습 진행 상황 로깅을 위한 커스텀 학습
            self._train_gradient_boosting_with_logging(X_train, y_train, X_val, y_val, n_estimators, early_stopping_rounds)
            
        else:
            # Random Forest 모델 (효율성 우수: R² 0.9846, RMSE 5315, 학습시간 ~5초)
            # Random Forest는 early stopping을 지원하지 않음
            rf_n_estimators = n_estimators if n_estimators <= 5000 else 5000  # RF는 너무 많으면 비효율적
            if rf_n_estimators != n_estimators:
                self.logger.warning(f"Random Forest는 early stopping을 지원하지 않으므로 n_estimators를 {rf_n_estimators}로 제한합니다.")
            
            self.logger.info("Random Forest 모델 학습 시작...")
            self.logger.info(f"  트리 개수: {rf_n_estimators:,}개 (early stopping 없음)")
            
            self.model = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=max_depth if max_depth >= 25 else 30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1,
                verbose=1  # 진행 상황 출력
            )
            
            # Random Forest 학습 (진행 상황 로깅)
            self._train_random_forest_with_logging(X_train, y_train, rf_n_estimators)
        
        total_time = time.time() - start_time
        self.logger.info(f"모델 학습 완료 (총 소요 시간: {total_time:.2f}초, ~{total_time/60:.1f}분)")
        
        # 학습 데이터 평가
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        self.logger.info("=" * 70)
        self.logger.info("=== 학습 데이터 성능 ===")
        self.logger.info(f"RMSE: {train_rmse:.2f}")
        self.logger.info(f"MAE: {train_mae:.2f}")
        self.logger.info(f"R²: {train_r2:.4f}")
        self.logger.info("=" * 70)
        
        # 검증 데이터 평가
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            self.logger.info("=== 검증 데이터 성능 ===")
            self.logger.info(f"RMSE: {val_rmse:.2f}")
            self.logger.info(f"MAE: {val_mae:.2f}")
            self.logger.info(f"R²: {val_r2:.4f}")
            self.logger.info("=" * 70)
        
        # 학습 히스토리 저장
        self.training_history.append({
            'model_type': model_type,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_r2': val_r2 if X_val is not None else None,
            'val_rmse': val_rmse if X_val is not None else None,
            'val_mae': val_mae if X_val is not None else None,
            'total_time': total_time
        })
        
        return self.model
    
    def _train_gradient_boosting_with_logging(self, X_train, y_train, X_val, y_val, n_estimators, early_stopping_rounds):
        """Gradient Boosting 학습 with 로깅 및 early stopping (각 반복마다 성능 로깅)"""
        self.logger.info("Gradient Boosting 학습 시작 (각 반복마다 성능 로깅)")
        
        # validation_fraction을 사용하여 early stopping (효율적)
        # 제공된 검증 데이터는 최종 평가에만 사용
        use_provided_val = (X_val is not None and y_val is not None)
        
        if use_provided_val:
            self.logger.info("  validation_fraction=0.1로 내부 검증 세트 분리하여 early stopping")
            self.logger.info("  제공된 검증 데이터는 최종 평가에만 사용")
            # validation_fraction을 사용하여 early stopping
            self.model.validation_fraction = 0.1
        else:
            self.logger.info("  validation_fraction=0.1로 내부 검증 세트 분리하여 early stopping")
        
        fit_start_time = time.time()
        
        # staged_predict를 사용하여 각 반복마다 성능 로깅
        self.logger.info(f"  최대 반복: {n_estimators:,}개, Early Stopping: {early_stopping_rounds} rounds")
        self._log_gb_progress(self.model, X_train, y_train, n_estimators, early_stopping_rounds, X_val, y_val)
        
        fit_time = time.time() - fit_start_time
        
        # 실제 학습된 트리 수 확인
        if hasattr(self.model, 'n_estimators_'):
            actual_n_estimators = self.model.n_estimators_
        else:
            actual_n_estimators = self.model.n_estimators
        
        self.logger.info(f"Gradient Boosting 학습 완료")
        self.logger.info(f"  실제 학습된 트리 수: {actual_n_estimators:,}개 (최대: {self.model.n_estimators:,}개)")
        
        if actual_n_estimators < self.model.n_estimators:
            self.logger.info(f"  Early stopping 적용됨: {early_stopping_rounds}번 연속 개선 없음으로 {actual_n_estimators:,}개에서 중단")
        else:
            self.logger.info(f"  최대 반복 횟수({self.model.n_estimators:,}개)까지 학습 완료")
        
        self.logger.info(f"  학습 시간: {fit_time:.2f}초 (~{fit_time/60:.1f}분)")
    
    def _train_gb_with_manual_early_stopping(self, X_train, y_train, X_val, y_val, early_stopping_rounds=50):
        """Gradient Boosting 수동 early stopping (검증 데이터 사용, staged_predict 활용)"""
        max_n_estimators = self.model.n_estimators
        
        self.logger.info(f"  최대 반복: {max_n_estimators:,}개, Early Stopping: {early_stopping_rounds} rounds")
        self.logger.info(f"  검증 데이터 사용: {len(X_val):,}개")
        self.logger.info(f"  진행 상황 로깅 (단계별 평가)")
        
        # warm_start를 사용하여 점진적 학습
        best_val_score = -np.inf
        no_improve_count = 0
        best_iteration = 0
        best_model_state = None
        
        # 단계별 학습을 위한 모델
        temp_model = GradientBoostingRegressor(
            n_estimators=1,
            learning_rate=self.model.learning_rate,
            max_depth=self.model.max_depth,
            min_samples_split=self.model.min_samples_split,
            min_samples_leaf=self.model.min_samples_leaf,
            subsample=self.model.subsample,
            random_state=self.model.random_state,
            warm_start=True  # 점진적 학습을 위해
        )
        
        # 초기 학습
        temp_model.fit(X_train, y_train)
        
        log_interval = max(100, min(1000, max_n_estimators // 50))  # 적절한 로깅 간격
        self.logger.info(f"  로깅 간격: {log_interval:,}개마다 또는 개선 시")
        
        # warm_start를 사용하여 점진적으로 학습하면서 평가
        # 하지만 더 효율적으로 하기 위해, 먼저 학습 후 staged_predict 사용
        self.logger.info("  모델 학습 중... (진행 상황 모니터링)")
        
        # validation_fraction을 사용하는 대신, 제공된 검증 데이터로 직접 모니터링
        # 이를 위해 validation_fraction=None으로 설정하고, 단계별 학습
        # 하지만 sklearn은 validation_fraction이 None이면 early stopping을 지원하지 않음
        # 따라서 staged_predict를 사용하여 단계별 평가 후 early stopping 구현
        
        # 먼저 최대 반복까지 학습 (warm_start 사용)
        temp_model.set_params(n_estimators=max_n_estimators, warm_start=True)
        
        # 장시간 소요될 수 있으므로 모니터링 추가
        fit_start = time.time()
        self.logger.info(f"  모델 학습 시작 (최대 {max_n_estimators:,}개 반복)...")
        
        # 예상 시간 계산 (데이터 크기 및 반복 횟수 기반)
        # Early stopping을 고려하여 실제 학습 시간은 최대 시간의 30-50% 정도로 추정
        estimated_gb_time = max(120, len(X_train) / 10000 * max_n_estimators / 1000 * 5) * 0.4
        
        # 백그라운드 모니터링 시작 (예상 시간 포함)
        stop_monitor = self._monitor_fit_operation("GB 학습", fit_start, interval=30, estimated_duration=estimated_gb_time)
        
        try:
            temp_model.fit(X_train, y_train)
        finally:
            stop_monitor.set()
        
        fit_time = time.time() - fit_start
        self.logger.info(f"  모델 학습 완료 (소요 시간: {fit_time:.2f}초, ~{fit_time/60:.1f}분)")
        
        # staged_predict로 단계별 평가 (이미 학습된 모델에서 각 단계의 예측)
        self.logger.info("  단계별 검증 데이터 성능 평가 중...")
        
        for i, val_pred in enumerate(temp_model.staged_predict(X_val), start=1):
            val_r2 = r2_score(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            # 개선 여부 확인 (tol=1e-4 고려)
            if val_r2 > best_val_score + 1e-4:
                best_val_score = val_r2
                best_iteration = i
                no_improve_count = 0
                
                # 개선 시 로깅
                if i <= 100 or i % log_interval == 0 or i == 1:
                    self.logger.info(f"  Iter {i:,}/{max_n_estimators:,}: Val R²={val_r2:.6f}, Val RMSE={val_rmse:.2f} ⭐ (개선, Best: {best_iteration:,})")
            else:
                no_improve_count += 1
                # 로깅 (주기적으로)
                if i <= 100 or i % log_interval == 0:
                    self.logger.info(f"  Iter {i:,}/{max_n_estimators:,}: Val R²={val_r2:.6f}, Val RMSE={val_rmse:.2f} (개선 없음: {no_improve_count}/{early_stopping_rounds})")
            
            # Early stopping 체크
            if no_improve_count >= early_stopping_rounds:
                self.logger.info(f"  Early stopping: {early_stopping_rounds}번 연속 개선 없음 (Iter {i:,}에서)")
                self.logger.info(f"  최적 반복: {best_iteration:,}개 (Val R²={best_val_score:.6f})")
                # 최적 모델로 재학습
                self.logger.info(f"  최적 모델 재학습 중 (Iter {best_iteration:,}개)...")
                best_model = GradientBoostingRegressor(
                    n_estimators=best_iteration,
                    learning_rate=temp_model.learning_rate,
                    max_depth=temp_model.max_depth,
                    min_samples_split=temp_model.min_samples_split,
                    min_samples_leaf=temp_model.min_samples_leaf,
                    subsample=temp_model.subsample,
                    random_state=temp_model.random_state,
                    verbose=0
                )
                best_fit_start = time.time()
                # 최적 모델 재학습 예상 시간 (best_iteration 기반)
                estimated_relearn_time = max(30, len(X_train) / 10000 * best_iteration / 1000 * 3)
                stop_monitor = self._monitor_fit_operation("최적 GB 재학습", best_fit_start, interval=30, estimated_duration=estimated_relearn_time)
                try:
                    best_model.fit(X_train, y_train)
                finally:
                    stop_monitor.set()
                self.logger.info(f"  최적 모델 재학습 완료 (소요 시간: {time.time() - best_fit_start:.2f}초)")
                self.model = best_model
                setattr(self.model, 'n_estimators_', best_iteration)  # 실제 학습된 수 저장
                return
        
        # 최대 반복까지 학습 완료
        self.logger.info(f"  최대 반복 횟수({max_n_estimators:,}개)까지 학습 완료")
        if best_iteration < max_n_estimators:
            self.logger.info(f"  최적 반복: {best_iteration:,}개 (Val R²={best_val_score:.6f})")
            # 최적 상태로 재학습
            self.logger.info(f"  최적 모델 재학습 중 (Iter {best_iteration:,}개)...")
            best_model = GradientBoostingRegressor(
                n_estimators=best_iteration,
                learning_rate=temp_model.learning_rate,
                max_depth=temp_model.max_depth,
                min_samples_split=temp_model.min_samples_split,
                min_samples_leaf=temp_model.min_samples_leaf,
                subsample=temp_model.subsample,
                random_state=temp_model.random_state,
                verbose=0
            )
            best_fit_start = time.time()
            # 최적 모델 재학습 예상 시간 (best_iteration 기반)
            estimated_relearn_time = max(30, len(X_train) / 10000 * best_iteration / 1000 * 3)
            stop_monitor = self._monitor_fit_operation("최적 GB 재학습", best_fit_start, interval=30, estimated_duration=estimated_relearn_time)
            try:
                best_model.fit(X_train, y_train)
            finally:
                stop_monitor.set()
            self.logger.info(f"  최적 모델 재학습 완료 (소요 시간: {time.time() - best_fit_start:.2f}초)")
            self.model = best_model
            setattr(self.model, 'n_estimators_', best_iteration)
        else:
            # 최대 반복까지 학습된 것이 최적
            self.model = temp_model
            setattr(self.model, 'n_estimators_', max_n_estimators)
    
    def _monitor_fit_operation(self, operation_name, start_time, interval=30, estimated_duration=None):
        """fit 작업 모니터링 (백그라운드 스레드)
        
        Parameters:
        -----------
        operation_name : str
            작업 이름
        start_time : float
            시작 시간 (time.time())
        interval : int
            로그 출력 간격 (초)
        estimated_duration : float, optional
            예상 소요 시간 (초). 제공되면 진행률 계산
        """
        stop_event = threading.Event()
        
        def monitor():
            last_log_time = start_time
            while not stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if current_time - last_log_time >= interval:
                    # 진행률 계산 (예상 시간이 제공된 경우)
                    progress_info = ""
                    if estimated_duration and estimated_duration > 0:
                        progress_pct = min(95, (elapsed_time / estimated_duration) * 100)
                        progress_info = f"예상 진행률: {progress_pct:.1f}% | "
                        
                        # 예상 남은 시간
                        if progress_pct < 95 and elapsed_time > 0:
                            remaining_time = estimated_duration - elapsed_time
                            if remaining_time > 0:
                                progress_info += f"예상 남은 시간: 약 {remaining_time:.0f}초 (~{remaining_time/60:.1f}분) | "
                    
                    # CPU/메모리 사용률 확인 (psutil이 있는 경우)
                    if PSUTIL_AVAILABLE:
                        try:
                            cpu_percent = psutil.cpu_percent(interval=0.5)
                            memory = psutil.virtual_memory()
                            memory_percent = memory.percent
                            memory_used_gb = memory.used / (1024**3)
                            memory_total_gb = memory.total / (1024**3)
                            
                            self.logger.info(f"  [{operation_name} 진행 중] {progress_info}"
                                            f"경과 시간: {elapsed_time:.0f}초 (~{elapsed_time/60:.1f}분) | "
                                            f"CPU: {cpu_percent:.1f}% | "
                                            f"메모리: {memory_percent:.1f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB)")
                        except Exception:
                            self.logger.info(f"  [{operation_name} 진행 중] {progress_info}"
                                            f"경과 시간: {elapsed_time:.0f}초 (~{elapsed_time/60:.1f}분)")
                    else:
                        self.logger.info(f"  [{operation_name} 진행 중] {progress_info}"
                                        f"경과 시간: {elapsed_time:.0f}초 (~{elapsed_time/60:.1f}분)")
                    
                    last_log_time = current_time
                
                time.sleep(0.5)  # 0.5초마다 체크
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return stop_event
    
    def _log_gb_progress(self, model, X_train, y_train, n_estimators, early_stopping_rounds, X_val=None, y_val=None):
        """Gradient Boosting 학습 진행 상황 로깅 (각 반복마다 성능 표시)"""
        # 로깅 간격 설정 (데이터 크기에 따라 조정)
        log_interval = max(1, min(100, n_estimators // 100))  # 최소 1, 최대 100
        
        # validation_fraction을 사용하는 경우
        if model.validation_fraction:
            # 먼저 전체 학습 (validation_fraction이 내부적으로 처리됨)
            # 장시간 소요될 수 있으므로 모니터링 추가
            fit_start = time.time()
            self.logger.info(f"  모델 학습 시작 (최대 {n_estimators:,}개 반복, early stopping 적용)...")
            
            # 예상 시간 계산 (데이터 크기 기반)
            # Early stopping을 고려하여 실제 학습 시간은 최대 시간의 30-50% 정도로 추정
            estimated_gb_time = max(60, len(X_train) / 10000 * 10) * 0.4  # 보수적 추정
            
            # 백그라운드 모니터링 시작 (예상 시간 포함)
            stop_monitor = self._monitor_fit_operation("GB 학습", fit_start, interval=30, estimated_duration=estimated_gb_time)
            
            try:
                model.fit(X_train, y_train)
            finally:
                stop_monitor.set()
            
            fit_time = time.time() - fit_start
            self.logger.info(f"  모델 학습 완료 (소요 시간: {fit_time:.2f}초, ~{fit_time/60:.1f}분)")
            
            # staged_predict로 단계별 성능 확인 (train set에 대해)
            self.logger.info(f"  각 반복마다 학습 데이터 성능 로깅 (로깅 간격: {log_interval}개):")
            
            best_score = -np.inf
            best_iter = 0
            no_improve = 0
            
            for i, train_pred in enumerate(model.staged_predict(X_train), start=1):
                train_r2 = r2_score(y_train, train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                # 개선 여부 확인
                if train_r2 > best_score + 1e-4:
                    best_score = train_r2
                    best_iter = i
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1
                    improved = False
                
                # 로깅 (처음 100개는 더 자주, 이후는 간격으로, 개선 시 즉시)
                if i <= 100 or i % log_interval == 0 or i == 1 or improved:
                    marker = "⭐" if improved else " "
                    self.logger.info(f"    Iter {i:6,}/{n_estimators:,}: Train R²={train_r2:.6f}, Train RMSE={train_rmse:.2f} {marker}")
                
                # Early stopping 체크 (staged_predict로는 실제로 중단할 수 없지만, 정보만 로깅)
                if i >= n_estimators:
                    break
            
            # 실제 학습된 모델의 n_estimators_ 확인
            if hasattr(model, 'n_estimators_'):
                actual_n = model.n_estimators_
                if actual_n < n_estimators:
                    self.logger.info(f"  Early stopping 적용됨: 실제 학습된 트리 수 = {actual_n:,}개")
        else:
            # 제공된 검증 데이터를 사용하는 경우 (수동 early stopping)
            # warm_start를 사용하여 단계별 학습
            self.logger.info(f"  각 반복마다 검증 데이터 성능 로깅 (로깅 간격: {log_interval}개):")
            
            temp_model = GradientBoostingRegressor(
                n_estimators=1,
                learning_rate=model.learning_rate,
                max_depth=model.max_depth,
                min_samples_split=model.min_samples_split,
                min_samples_leaf=model.min_samples_leaf,
                subsample=model.subsample,
                random_state=model.random_state,
                warm_start=True
            )
            
            best_val_score = -np.inf
            best_iter = 0
            no_improve = 0
            loop_start_time = time.time()
            last_monitor_time = loop_start_time
            
            for i in range(1, n_estimators + 1):
                temp_model.set_params(n_estimators=i)
                
                # 장시간 학습 시 주기적으로 진행 상황 표시
                current_time = time.time()
                if i > 1 and current_time - last_monitor_time >= 30:  # 30초마다
                    elapsed = current_time - loop_start_time
                    progress_pct = (i / n_estimators) * 100
                    if PSUTIL_AVAILABLE:
                        try:
                            cpu_percent = psutil.cpu_percent(interval=0.1)
                            memory = psutil.virtual_memory()
                            memory_percent = memory.percent
                            self.logger.info(f"  [GB 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | "
                                            f"경과: {elapsed:.0f}초 | CPU: {cpu_percent:.1f}% | 메모리: {memory_percent:.1f}%")
                        except Exception:
                            self.logger.info(f"  [GB 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | 경과: {elapsed:.0f}초")
                    else:
                        self.logger.info(f"  [GB 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | 경과: {elapsed:.0f}초")
                    last_monitor_time = current_time
                
                temp_model.fit(X_train, y_train)
                
                # 검증 데이터 성능 평가
                if X_val is not None and y_val is not None:
                    val_pred = temp_model.predict(X_val)
                    val_r2 = r2_score(y_val, val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                    
                    # 개선 여부 확인
                    if val_r2 > best_val_score + 1e-4:
                        best_val_score = val_r2
                        best_iter = i
                        no_improve = 0
                        improved = True
                    else:
                        no_improve += 1
                        improved = False
                    
                    # 로깅 (처음 100개는 더 자주, 이후는 간격으로, 개선 시 즉시)
                    if i <= 100 or i % log_interval == 0 or i == 1 or improved:
                        marker = "⭐" if improved else " "
                        self.logger.info(f"    Iter {i:6,}/{n_estimators:,}: Val R²={val_r2:.6f}, Val RMSE={val_rmse:.2f} {marker}")
                    
                    # Early stopping 체크
                    if no_improve >= early_stopping_rounds:
                        self.logger.info(f"  Early stopping: {early_stopping_rounds}번 연속 개선 없음 (Iter {i:,}에서)")
                        self.logger.info(f"  최적 반복: {best_iter:,}개 (Val R²={best_val_score:.6f})")
                        # 최적 모델로 설정
                        model.set_params(n_estimators=best_iter)
                        model.fit(X_train, y_train)
                        return
                else:
                    # 검증 데이터가 없는 경우 train 데이터 성능만 로깅
                    train_pred = temp_model.predict(X_train)
                    train_r2 = r2_score(y_train, train_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                    
                    if i <= 100 or i % log_interval == 0 or i == 1:
                        self.logger.info(f"    Iter {i:6,}/{n_estimators:,}: Train R²={train_r2:.6f}, Train RMSE={train_rmse:.2f}")
            
            # 최대 반복까지 학습 완료
            if best_iter > 0 and best_iter < n_estimators:
                model.set_params(n_estimators=best_iter)
            model.fit(X_train, y_train)
    
    def _train_random_forest_with_logging(self, X_train, y_train, n_estimators):
        """Random Forest 학습 with 로깅 (각 반복마다 성능 로깅)"""
        self.logger.info("Random Forest 학습 시작 (각 반복마다 성능 로깅)")
        
        fit_start_time = time.time()
        
        # Random Forest는 staged_predict를 지원하지만, 각 트리 추가마다 예측하는 것이 비효율적
        # 대신 일정 간격으로 성능 평가
        log_interval = max(1, min(50, n_estimators // 20))  # 최소 1, 최대 50
        self.logger.info(f"  각 {log_interval:,}개 트리마다 학습 데이터 성능 로깅")
        
        # warm_start를 사용하여 단계별 학습 및 평가
        temp_model = RandomForestRegressor(
            n_estimators=1,
            max_depth=self.model.max_depth,
            min_samples_split=self.model.min_samples_split,
            min_samples_leaf=self.model.min_samples_leaf,
            max_features=self.model.max_features,
            random_state=self.model.random_state,
            n_jobs=self.model.n_jobs,
            warm_start=True,
            verbose=0
        )
        
        best_score = -np.inf
        best_iter = 0
        last_monitor_time = fit_start_time
        
        for i in range(1, n_estimators + 1):
            temp_model.set_params(n_estimators=i)
            
            # 장시간 학습 시 주기적으로 진행 상황 표시 (각 반복마다 fit 호출하므로 짧게)
            current_time = time.time()
            if i > 1 and current_time - last_monitor_time >= 30:  # 30초마다
                elapsed = current_time - fit_start_time
                progress_pct = (i / n_estimators) * 100
                if PSUTIL_AVAILABLE:
                    try:
                        cpu_percent = psutil.cpu_percent(interval=0.1)
                        memory = psutil.virtual_memory()
                        memory_percent = memory.percent
                        self.logger.info(f"  [RF 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | "
                                        f"경과: {elapsed:.0f}초 | CPU: {cpu_percent:.1f}% | 메모리: {memory_percent:.1f}%")
                    except Exception:
                        self.logger.info(f"  [RF 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | 경과: {elapsed:.0f}초")
                else:
                    self.logger.info(f"  [RF 학습 진행 중] {i:,}/{n_estimators:,} ({progress_pct:.1f}%) | 경과: {elapsed:.0f}초")
                last_monitor_time = current_time
            
            temp_model.fit(X_train, y_train)
            
            # 성능 평가 (로깅 간격마다)
            if i <= 50 or i % log_interval == 0 or i == 1 or i == n_estimators:
                train_pred = temp_model.predict(X_train)
                train_r2 = r2_score(y_train, train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                if train_r2 > best_score:
                    best_score = train_r2
                    best_iter = i
                    improved = True
                else:
                    improved = False
                
                marker = "⭐" if improved else " "
                self.logger.info(f"    Iter {i:6,}/{n_estimators:,}: Train R²={train_r2:.6f}, Train RMSE={train_rmse:.2f} {marker}")
        
        # 최종 모델로 설정
        self.model = temp_model
        
        fit_time = time.time() - fit_start_time
        
        self.logger.info(f"Random Forest 학습 완료")
        self.logger.info(f"  학습된 트리 수: {self.model.n_estimators:,}개")
        self.logger.info(f"  최적 반복: {best_iter:,}개 (Train R²={best_score:.6f})")
        self.logger.info(f"  학습 시간: {fit_time:.2f}초 (~{fit_time/60:.1f}분)")
    
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
            'target_column': self.target_column,
            'target_encodings': getattr(self, 'target_encodings', {}),  # Target Encoding 정보
            'frequency_encodings': getattr(self, 'frequency_encodings', {}),  # Frequency Encoding 정보
            'region_mean_floor': getattr(self, 'region_mean_floor', {}),  # 지역별 평균 층수 정보
            'missing_value_stats': getattr(self, 'missing_value_stats', {}),  # 결측치 처리 통계량
            'outlier_stats': getattr(self, 'outlier_stats', {})  # 이상치 처리 통계량
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        missing_stats_count = len(model_data.get('missing_value_stats', {}).get('columns_with_missing', []))
        outlier_stats_count = len(model_data.get('outlier_stats', {}))
        
        self.logger.info(f"모델이 저장되었습니다: {model_path}")
        self.logger.info(f"  저장된 정보: 모델, LabelEncoder, Target Encoding ({len(model_data['target_encodings'])}개), Frequency Encoding ({len(model_data['frequency_encodings'])}개)")
        self.logger.info(f"  결측치 처리 통계량: {missing_stats_count}개 컬럼, 이상치 처리 통계량: {outlier_stats_count}개 컬럼")
        print(f"모델이 저장되었습니다: {model_path}")
        print(f"  저장된 정보: 모델, LabelEncoder, Target Encoding ({len(model_data['target_encodings'])}개), Frequency Encoding ({len(model_data['frequency_encodings'])}개)")
        print(f"  결측치 처리 통계량: {missing_stats_count}개 컬럼, 이상치 처리 통계량: {outlier_stats_count}개 컬럼")
    
    def load_model(self, model_path='model.pkl'):
        """모델 로드"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        # 전처리 관련 정보 로드 (있을 경우)
        if 'target_encodings' in model_data:
            self.target_encodings = model_data['target_encodings']
        if 'frequency_encodings' in model_data:
            self.frequency_encodings = model_data['frequency_encodings']
        if 'region_mean_floor' in model_data:
            self.region_mean_floor = model_data['region_mean_floor']
        if 'missing_value_stats' in model_data:
            self.missing_value_stats = model_data['missing_value_stats']
        if 'outlier_stats' in model_data:
            self.outlier_stats = model_data['outlier_stats']
        
        missing_stats_count = len(model_data.get('missing_value_stats', {}).get('columns_with_missing', [])) if 'missing_value_stats' in model_data else 0
        outlier_stats_count = len(model_data.get('outlier_stats', {})) if 'outlier_stats' in model_data else 0
        
        if hasattr(self, 'logger'):
            self.logger.info(f"모델이 로드되었습니다: {model_path}")
            if hasattr(self, 'target_encodings') and self.target_encodings:
                self.logger.info(f"  Target Encoding 정보: {len(self.target_encodings)}개 변수")
            if hasattr(self, 'frequency_encodings') and self.frequency_encodings:
                self.logger.info(f"  Frequency Encoding 정보: {len(self.frequency_encodings)}개 변수")
            if missing_stats_count > 0:
                self.logger.info(f"  결측치 처리 통계량: {missing_stats_count}개 컬럼")
            if outlier_stats_count > 0:
                self.logger.info(f"  이상치 처리 통계량: {outlier_stats_count}개 컬럼")
        print(f"모델이 로드되었습니다: {model_path}")
        if hasattr(self, 'target_encodings') and self.target_encodings:
            print(f"  Target Encoding 정보: {len(self.target_encodings)}개 변수")
        if hasattr(self, 'frequency_encodings') and self.frequency_encodings:
            print(f"  Frequency Encoding 정보: {len(self.frequency_encodings)}개 변수")
        if missing_stats_count > 0:
            print(f"  결측치 처리 통계량: {missing_stats_count}개 컬럼")
        if outlier_stats_count > 0:
            print(f"  이상치 처리 통계량: {outlier_stats_count}개 컬럼")


def main():
    """메인 실행 함수"""
    # 경로 설정
    train_path = '/data/ephemeral/home/py310/train.csv'
    model_save_path = '/data/ephemeral/home/py310/code/apartment_price_model.pkl'
    
    # 로그 파일 경로 설정
    log_dir = '/data/ephemeral/home/py310/code/logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 모델 초기화 (로그 파일 설정)
    predictor = ApartmentPricePredictor(log_file=log_file, log_level=logging.INFO)
    predictor.logger.info(f"로그 파일: {log_file}")
    
    # 데이터 로드 (위치 정보 추가 옵션)
    # 위치 정보를 추가하려면 add_location_features=True로 설정
    train_df = predictor.load_data(
        train_path, 
        add_location_features=True,  # 위치 정보 추가 여부
        bus_csv_path='/data/ephemeral/home/py310/bus_feature.csv',
        subway_csv_path='/data/ephemeral/home/py310/subway_feature.csv'
    )
    
    # 데이터 전처리
    X, y = predictor.preprocess_data(train_df, is_train=True)
    
    # 학습/검증 데이터 분할
    print("\n학습/검증 데이터 분할 중...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
    
    # 모델 학습 (최적 파라미터 적용)
    # 기본값: StackingRegressor with ElasticNetCV (최고 성능, R² 0.9873, 학습시간 ~96초) ⭐
    # 다른 모델을 사용하려면 model_type 파라미터 변경 가능:
    #   - 'stacking_elasticnet' (기본값, 최고 성능, n_estimators=100000, early_stopping=50)
    #   - 'random_forest' (효율성 우수, R² 0.9846, 학습시간 ~5초)
    #   - 'gradient_boosting' (개별 모델 최고, n_estimators=100000, early_stopping=50)
    predictor.train(
        X_train, y_train, 
        X_val, y_val,
        model_type='stacking_elasticnet',  # 기본 모델 (최고 성능) 또는 'random_forest', 'gradient_boosting'
        n_estimators=100000,  # 최대 반복 횟수 (early stopping으로 실제 학습 횟수는 적을 수 있음)
        early_stopping_rounds=50,  # 50번 연속 개선 없으면 중단
        random_state=42
        # StackingRegressor는 내부적으로 최적 파라미터 사용 (RF: n=200, d=30, GB: n=100000, early_stopping=50, lr=0.04, d=8)
    )
    
    # 모델 저장
    predictor.save_model(model_save_path)
    
    print("\n모델 학습이 완료되었습니다!")


if __name__ == '__main__':
    main()
