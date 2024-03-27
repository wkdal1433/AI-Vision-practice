import random
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

import warnings
warnings.filterwarnings(action='ignore')

# 랜덤 시드 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # 시드 고정

# 데이터 불러오기
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# 누락된 값이 있는 열 찾기
train_missing_cols = train_df.columns[train_df.isna().any()].tolist()
test_missing_cols = test_df.columns[test_df.isna().any()].tolist()

# 훈련 및 테스트 데이터에서 누락된 열 결합
all_missing_cols = list(set(train_missing_cols + test_missing_cols))

# 평균으로 누락된 값을 채우기
for col in all_missing_cols:
    col_mean = train_df[col].mean()
    train_df[col].fillna(col_mean, inplace=True)
    test_df[col].fillna(col_mean, inplace=True)

# 시계열 특성 추가
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])
train_df['dayofweek'] = train_df['일시'].dt.dayofweek
train_df['is_weekend'] = train_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
train_df['is_holiday'] = train_df['일시'].apply(lambda x: 1 if x in holiday_dates else 0)
test_df['dayofweek'] = test_df['일시'].dt.dayofweek
test_df['is_weekend'] = test_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
test_df['is_holiday'] = test_df['일시'].apply(lambda x: 1 if x in holiday_dates else 0)

# 특성 엔지니어링
train_df['temp_mul_rain'] = train_df['기온(°C)'] * train_df['강수량(mm)']
test_df['temp_mul_rain'] = test_df['기온(°C)'] * test_df['강수량(mm)']

# 이상치 처리
train_df = train_df[train_df['전력소비량(kWh)'] < train_df['전력소비량(kWh)'].quantile(0.99)]

# Train Data Pre-Processing
train_x = train_df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

# 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)

# 특성 선택
selector = SelectKBest(score_func=f_regression, k=10)
train_x_selected = selector.fit_transform(train_x_scaled, train_y)

# Regression Model Fit
model = RandomForestRegressor()
model.fit(train_x_selected, train_y)

# Test Data Pre-Processing
test_df['month'] = test_df['일시'].apply(lambda x: int(x.strftime("%m")))
test_df['day'] = test_df['일시'].apply(lambda x: int(x.strftime("%d")))
test_df['time'] = test_df['일시'].apply(lambda x: int(x.strftime("%H")))

test_x = test_df.drop(columns=['num_date_time', '일시'])

# 스케일링 및 특성 선택 적용
test_x_scaled = scaler.transform(test_x)
test_x_selected = selector.transform(test_x_scaled)

# Inference
preds = model.predict(test_x_selected)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = preds
submission.to_csv('./baseline_submission2.csv', index=False)