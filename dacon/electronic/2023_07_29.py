import random
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

import warnings
warnings.filterwarnings(action='ignore')

# 고정된 랜덤 시드 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)  # 시드 고정

# 데이터 로드
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
building_info_df = pd.read_csv('./building_info.csv')

# '일시' 컬럼을 datetime 형식으로 변환
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

# 'num_date_time' 컬럼 삭제
train_df.drop(columns=['num_date_time'], inplace=True)
test_df.drop(columns=['num_date_time'], inplace=True)

# '전력소비량(kWh)' 컬럼을 미리 NaN 값으로 채우기
test_df['전력소비량(kWh)'] = np.nan

# 데이터 전처리 및 Prophet 모델 사용
def preprocess_and_predict(train_df, test_df):
    # 각 컬럼의 결측치를 앞선 값으로 채움
    train_df.fillna(method='ffill', inplace=True)
    test_df.fillna(method='ffill', inplace=True)

    # Prophet 모델 피팅을 위해 필요한 데이터 프레임 구성
    prophet_train_df = train_df[['일시', '전력소비량(kWh)']]
    prophet_train_df.rename(columns={'일시': 'ds', '전력소비량(kWh)': 'y'}, inplace=True)
    
    # Prophet 모델 정의 및 학습
    prophet_model = Prophet()
    prophet_model.fit(prophet_train_df)
    
    # Test 데이터를 활용하여 예측 수행
    prophet_test_df = test_df[['일시', '전력소비량(kWh)']]
    prophet_test_df.rename(columns={'일시': 'ds', '전력소비량(kWh)': 'y'}, inplace=True)
    try:
        prophet_preds = prophet_model.predict(prophet_test_df)
        test_df['전력소비량(kWh)'] = prophet_preds['yhat'].values
    except ValueError:
        print("ValueError occurred during Prophet prediction. Check your data or hyperparameters.")
    
    return test_df

# 앙상블을 위한 예측 결과 조합
def ensemble_predictions(test_df, preds_rf):
    # Prophet과 RandomForestRegressor 예측 결과를 앙상블
    final_preds = 0.5 * test_df['전력소비량(kWh)'] + 0.5 * preds_rf
    return final_preds

# 특성 스케일링
def feature_scaling(train_df, test_df):
    # Min-Max Scaling을 위한 객체 생성 및 학습 데이터에 적용
    scaler = MinMaxScaler()
    train_x_scaled = scaler.fit_transform(train_df.drop(columns=['전력소비량(kWh)']))
    train_df_scaled = pd.DataFrame(train_x_scaled, columns=train_df.drop(columns=['전력소비량(kWh)']).columns)
    
    # 테스트 데이터에도 동일한 스케일러 적용
    test_x_scaled = scaler.transform(test_df.drop(columns=['전력소비량(kWh)']))
    test_df_scaled = pd.DataFrame(test_x_scaled, columns=test_df.drop(columns=['전력소비량(kWh)']).columns)
    
    return train_df_scaled, test_df_scaled

# Hyperparameter Tuning for RandomForestRegressor
def rf_hyperparameter_tuning(train_x, train_y):
    # RandomForestRegressor의 하이퍼파라미터 튜닝
    # 여기에서 하이퍼파라미터 튜닝을 구현합니다. (예: 그리드 서치 또는 랜덤 서치 사용)
    # 예시를 위해 기본 파라미터로 RandomForestRegressor를 사용합니다.
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    return model

# Train Data Pre-Processing
train_df.fillna(train_df.mean(), inplace=True)

train_x = train_df.drop(columns=['일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

# Hyperparameter Tuning for RandomForestRegressor
rf_model = rf_hyperparameter_tuning(train_x, train_y)

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 월, 일, 시간으로 나눔
test_x = test_df.drop(columns=['일시', '전력소비량(kWh)'])

# 특성 스케일링 수행
train_df_scaled, test_x_scaled = feature_scaling(train_x, test_x)

# RandomForestRegressor로 예측
preds_rf = rf_model.predict(test_x_scaled)

# Prophet을 사용하여 예측 및 앙상블
final_preds = ensemble_predictions(test_df, preds_rf)

# 평가
actual_values = test_df['전력소비량(kWh)']
mse = mean_squared_error(actual_values, final_preds)
mae = mean_absolute_error(actual_values, final_preds)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = final_preds
submission.to_csv('./modified_baseline_submission.csv', index=False)