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
        self.region_mean_floor = {}  # 시군구별 평균 층수 (층수 지역정규화용, 모델 저장 시 포함)
        self.target_encodings = {}  # Target Encoding을 위한 타겟 평균 (예측 시 사용)
        self.frequency_encodings = {}  # Frequency Encoding을 위한 빈도 정보 (예측 시 사용)
        
    def load_model(self, model_path='apartment_price_model.pkl'):
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
        
        print(f"모델이 로드되었습니다: {model_path}")
    
    def preprocess_data(self, df):
        """데이터 전처리 (예측용)"""
        print("데이터 전처리 중...")
        X = df.copy()
        
        # 연립주택 데이터 제거 (k-단지분류(아파트,주상복합등등)가 "연립주택"인 경우)
        complex_type_col = 'k-단지분류(아파트,주상복합등등)'
        if complex_type_col in X.columns:
            before_count = len(X)
            # 연립주택 데이터 제거
            X = X[X[complex_type_col] != '연립주택'].copy()
            removed_count = before_count - len(X)
            
            if removed_count > 0:
                print(f"연립주택 데이터 제거: {removed_count:,}개 제거됨 (제거 전: {before_count:,}개, 제거 후: {len(X):,}개)")
        
        # 계약년월과 계약일을 합쳐서 YYMMDDhhmmss 형태의 파생변수 생성
        if '계약년월' in X.columns and '계약일' in X.columns:
            print("계약일시 파생변수 생성 중...")
            
            # 계약년월을 문자열로 변환
            contract_year_month = X['계약년월'].astype(str)
            
            # 연도 추출 (YYYYMM -> YYYY, MM)
            X['계약연도'] = contract_year_month.str[:4]
            X['계약월'] = contract_year_month.str[4:6]
            
            # 계약일을 2자리 문자열로 변환 (한 자리 수는 앞에 0 추가)
            X['계약일_포맷'] = X['계약일'].astype(int).astype(str).str.zfill(2)
            
            # YYMMDDhhmmss 형태: YYYY + MM + DD
            X['계약일자'] = (
                X['계약연도'].astype(str) + 
                X['계약월'].astype(str) + 
                X['계약일_포맷'].astype(str)
            ).astype(int)
            
            # 임시 컬럼 제거
            X = X.drop(columns=['계약연도', '계약월', '계약일_포맷'])
            
            print(f"계약일자 파생변수 생성 완료: 예시 = {X['계약일자'].iloc[0]}")
        
        # 전용면적을 평형대별 파생변수로 변환
        if '전용면적(㎡)' in X.columns:
            print("전용면적 평형대 파생변수 생성 중...")
            
            # 평형 계산 (1평 ≈ 3.3058㎡)
            X['평형'] = X['전용면적(㎡)'] / 3.3058
            
            # ㎡ 기준 평형대 분류
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
            X['평형대_카테고리'] = X['전용면적(㎡)'].apply(categorize_area_by_pyeong)
            
            # 평형대 숫자 인코딩 (학습 시와 동일한 매핑)
            pyeong_mapping = {
                'Unknown': 0,
                '소형_60이하': 1,
                '중소형_60_85': 2,
                '중형_85_102': 3,
                '중대형_102_135': 4,
                '대형_135초과': 5
            }
            X['평형대_코드'] = X['평형대_카테고리'].map(pyeong_mapping).fillna(0).astype(int)
            
            # 평형대별 더미 변수 생성 (선택적)
            X['평형대_소형'] = (X['평형대_코드'] == 1).astype(int)
            X['평형대_중소형'] = (X['평형대_코드'] == 2).astype(int)
            X['평형대_중형'] = (X['평형대_코드'] == 3).astype(int)
            X['평형대_중대형'] = (X['평형대_코드'] == 4).astype(int)
            X['평형대_대형'] = (X['평형대_코드'] == 5).astype(int)
            
            print(f"평형대 파생변수 생성 완료")
        
        # 건축년도 관련 파생변수 생성 (연식, 노후도) - 성능 향상 확인
        if '건축년도' in X.columns and '계약년월' in X.columns:
            # 계약년월에서 연도 추출
            contract_year = X['계약년월'].astype(str).str[:4].astype(int)
            
            # 연식 계산 (계약년도 - 건축년도)
            X['연식'] = (contract_year - X['건축년도']).clip(lower=0)
            
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
            
            X['노후도_코드'] = X['연식'].apply(categorize_age)
            
            # 노후도별 더미 변수 생성
            X['노후도_신축'] = (X['노후도_코드'] == 1).astype(int)
            X['노후도_준신축'] = (X['노후도_코드'] == 2).astype(int)
            X['노후도_중고'] = (X['노후도_코드'] == 3).astype(int)
            X['노후도_노후'] = (X['노후도_코드'] == 4).astype(int)
            X['노후도_구식'] = (X['노후도_코드'] == 5).astype(int)
            
            print(f"연식 파생변수 생성 완료")
        
        # 시군구 분리 파생변수 생성 (시, 구(군), 동으로 분리) - 학습 시와 동일하게 적용
        # Target/Frequency Encoding 적용 전에 생성해야 함
        if '시군구' in X.columns:
            print("시군구 분리 파생변수 생성 중...")
            
            # 시군구를 공백으로 분리 (형식: "서울특별시 강남구 개포동")
            sigungu_split = X['시군구'].str.split(' ', expand=True)
            
            # 시 추출 (첫 번째 요소)
            X['시'] = sigungu_split[0] if len(sigungu_split.columns) > 0 else ''
            X['시'] = X['시'].fillna('Unknown')
            
            # 구(군) 추출 (두 번째 요소)
            X['구_군'] = sigungu_split[1] if len(sigungu_split.columns) > 1 else ''
            X['구_군'] = X['구_군'].fillna('Unknown')
            
            # 동 추출 (세 번째 요소, 있으면)
            X['동'] = sigungu_split[2] if len(sigungu_split.columns) > 2 else ''
            X['동'] = X['동'].fillna('Unknown')
            
            print(f"시군구 분리 파생변수 생성 완료: 시({X['시'].nunique()}개), 구(군)({X['구_군'].nunique()}개), 동({X['동'].nunique()}개)")
        
        # Target Encoding 및 Frequency Encoding 적용 (예측 시)
        # Target Encoding
        if hasattr(self, 'target_encodings') and self.target_encodings:
            for col, encoding_map in self.target_encodings.items():
                if col in X.columns:
                    # 학습 시 사용된 평균값이 없으면 0 사용
                    default_value = 0
                    if hasattr(self, 'target_mean'):
                        default_value = self.target_mean
                    X[f'{col}_타겟평균'] = X[col].map(encoding_map).fillna(default_value)
            
            # 시군구 분리 파생변수에 대한 Target Encoding (구_군, 동)
            split_target_cols = ['구_군', '동']
            for col in split_target_cols:
                if col in X.columns and col in self.target_encodings:
                    encoding_map = self.target_encodings[col]
                    default_value = 0
                    if hasattr(self, 'target_mean'):
                        default_value = self.target_mean
                    X[f'{col}_타겟평균'] = X[col].map(encoding_map).fillna(default_value)
            
            print(f"Target Encoding 적용 완료 ({len(self.target_encodings)}개 변수)")
        
        # Frequency Encoding
        if hasattr(self, 'frequency_encodings') and self.frequency_encodings:
            for col, freq_map in self.frequency_encodings.items():
                if col in X.columns:
                    X[f'{col}_빈도'] = X[col].map(freq_map).fillna(0)
            
            # 시군구 분리 파생변수에 대한 Frequency Encoding (시, 구_군, 동)
            split_freq_cols = ['시', '구_군', '동']
            for col in split_freq_cols:
                if col in X.columns and col in self.frequency_encodings:
                    freq_map = self.frequency_encodings[col]
                    X[f'{col}_빈도'] = X[col].map(freq_map).fillna(0)
            
            print(f"Frequency Encoding 적용 완료 ({len(self.frequency_encodings)}개 변수)")
        
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
        existing_remove_cols = [col for col in columns_to_remove if col in X.columns]
        if existing_remove_cols:
            X = X.drop(columns=existing_remove_cols)
            print(f"불필요한 컬럼 제거 완료: {len(existing_remove_cols)}개 컬럼 제거됨")
        
        # 평형대 카테고리 컬럼은 이미 더미 변수로 변환되었으므로 제거
        if '평형대_카테고리' in X.columns:
            X = X.drop(columns=['평형대_카테고리'])
        
        # 로그 변환 변수 추가 (학습 시와 동일하게 적용)
        if '전용면적(㎡)' in X.columns:
            X['전용면적_log'] = np.log1p(X['전용면적(㎡)'].clip(lower=0))
        
        if '평형' in X.columns:
            X['평형_log'] = np.log1p(X['평형'].clip(lower=0))
        
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
        """CSV 파일에서 데이터를 읽어 예측
        
        Parameters:
        -----------
        csv_path : str
            입력 CSV 파일 경로
        output_path : str, optional
            출력 CSV 파일 경로 (sample_submission.csv 형식으로 저장)
            None이면 결과만 반환하고 저장하지 않음
            
        Returns:
        --------
        pandas.DataFrame
            예측 결과 DataFrame (target 컬럼만 포함, sample_submission.csv 형식)
        """
        print(f"데이터 로딩 중: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"데이터 shape: {df.shape}")
        
        # 원본 데이터의 인덱스와 연립주택 여부 정보 저장
        original_count = len(df)
        
        # 연립주택 여부 확인 (전처리 전)
        complex_type_col = 'k-단지분류(아파트,주상복합등등)'
        if complex_type_col in df.columns:
            연립주택_mask = df[complex_type_col] == '연립주택'
            연립주택_count = 연립주택_mask.sum()
        else:
            연립주택_mask = pd.Series([False] * len(df), index=df.index)
            연립주택_count = 0
        
        # 전처리 (연립주택 제거 등)
        X = self.preprocess_data(df)
        
        # 전처리 후 행 수 확인
        processed_count = len(X)
        if processed_count != original_count:
            removed_count = original_count - processed_count
            print(f"  전처리 과정에서 {removed_count}개 행이 제거되었습니다.")
        
        # 예측
        print("예측 중...")
        predictions = self.predict(X)
        
        # 원본 데이터와 동일한 행 수를 유지하기 위해 결과 생성
        # sample_submission.csv는 원본 test.csv의 모든 행에 대해 예측값을 제공해야 함
        result_predictions = np.zeros(original_count, dtype=float)
        
        if processed_count < original_count:
            # 연립주택이 제거된 경우: 예측 결과를 원본 순서에 맞게 매핑
            # 전처리 전 원본 DataFrame에서 연립주택이 아닌 행의 원본 위치 추적
            # X는 전처리 후 DataFrame이므로, 원본 df에서의 위치와 매핑 필요
            
            # 원본 df에서 연립주택이 아닌 행의 인덱스 (순서대로)
            valid_original_indices = df[~연립주택_mask].index.tolist()
            
            # 예측 결과를 원본 순서대로 할당
            for i, original_idx in enumerate(valid_original_indices):
                # 원본 DataFrame에서의 위치 (0부터 시작하는 정수 인덱스)
                original_position = df.index.get_loc(original_idx)
                if i < len(predictions):
                    result_predictions[original_position] = float(predictions[i])
            
            # 연립주택 행에 대한 처리 (평균값으로 채움)
            if len(predictions) > 0:
                fill_value = float(predictions.mean())
            else:
                fill_value = 0.0
            
            for original_idx in df[연립주택_mask].index:
                original_position = df.index.get_loc(original_idx)
                result_predictions[original_position] = fill_value
            
            if 연립주택_count > 0:
                print(f"  연립주택 행 {연립주택_count}개에 평균값 {fill_value:.0f}으로 채움")
        else:
            # 모든 행이 유효한 경우: 순서대로 할당 (원본 순서와 동일)
            result_predictions = predictions.astype(float)
        
        # sample_submission.csv 형식으로 결과 생성 (target 컬럼만 포함, 인덱스 제거)
        result_df = pd.DataFrame({
            'target': result_predictions
        })
        
        # 결과 저장 (sample_submission.csv와 동일한 형식)
        if output_path:
            # 인덱스 없이, 헤더만 포함하여 저장
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"예측 결과가 저장되었습니다: {output_path}")
            print(f"  형식: sample_submission.csv와 동일 (헤더: target, 예측값만 포함)")
            print(f"  총 {len(result_df)}개 예측값 저장됨")
            print(f"  예측 가격 범위: {predictions.min():.0f} ~ {predictions.max():.0f}")
        
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
    
    print(f"\n{'='*70}")
    print(f"예측 완료!")
    print(f"{'='*70}")
    print(f"총 {len(result_df)}개 데이터에 대한 예측이 완료되었습니다.")
    print(f"출력 파일 형식: sample_submission.csv와 동일")
    print(f"  - 헤더: target")
    print(f"  - 데이터: 예측값만 포함 (인덱스 없음)")
    print(f"예측 가격 범위: {result_df['target'].min():.0f} ~ {result_df['target'].max():.0f}")
    print(f"예측 가격 평균: {result_df['target'].mean():.2f}")
    print(f"예측 가격 중앙값: {result_df['target'].median():.2f}")


if __name__ == '__main__':
    main()
