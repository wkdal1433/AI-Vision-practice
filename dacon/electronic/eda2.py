import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np

train_df = pd.read_csv('./train.csv')
building_info = pd.read_csv('./building_info.csv')
test_df = pd.read_csv('./test.csv')

train_df = train_df.rename(columns={
    '건물번호': 'building_number',
    '일시': 'date_time',
    '기온(C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'windspeed',
    '습도(%)': 'humidity',
    '일조(hr)': 'sunshine',
    '일사(MJ/m2)': 'solar_radiation',
    '전력소비량(kWh)': 'power_consumption'
})
train_df.drop('num_date_time', axis=1, inplace=True)

test_df = test_df.rename(columns={
    '건물번호': 'building_number',
    '일시': 'date_time',
    '기온(C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'windspeed',
    '습도(%)': 'humidity',
    '일조(hr)': 'sunshine',
    '일사(MJ/m2)': 'solar_radiation',
    '전력소비량(kWh)': 'power_consumption'
})
test_df.drop('num_date_time', axis=1, inplace=True)

building_info = building_info.rename(columns={
    '건물번호': 'building_number',
    '건물유형': 'building_type',
    '연면적(m2)': 'total_area',
    '냉방면적(m2)': 'cooling_area',
    '태양광용량(kW)': 'solar_power_capacity',
    'ESS저장용량(kWh)': 'ess_capacity',
    'PCS용량(kW)': 'pcs_capacity'
})

translation_dict = {
    '건물기타': 'Other Buildings',
    '공공': 'Public',
    '대학교': 'University',
    '데이터센터': 'Data Center',
    '백화점및아울렛': 'Department Store and Outlet',
    '병원': 'Hospital',
    '상용': 'Commercial',
    '아파트': 'Apartment',
    '연구소': 'Research Institute',
    '지식산업센터': 'Knowledge Industry Center',
    '할인마트': 'Discount Mart',
    '호텔및리조트': 'Hotel and Resort'
}
building_info['building_type'] = building_info['building_type'].replace(translation_dict)

train_df = pd.merge(train_df, building_info, on='building_number', how='left')
test_df = pd.merge(test_df, building_info, on='building_number', how='left')

# Drop rows with missing solar_power_capacity, ess_capacity, and pcs_capacity values
train_df = train_df.dropna(subset=['solar_power_capacity', 'ess_capacity', 'pcs_capacity'])

train_df['date_time'] = pd.to_datetime(train_df['date_time'], format='%Y%m%d %H')

# date time feature 생성
train_df['hour'] = train_df['date_time'].dt.hour
train_df['day'] = train_df['date_time'].dt.day
train_df['month'] = train_df['date_time'].dt.month
train_df['year'] = train_df['date_time'].dt.year

# 예측 결과를 저장할 DataFrame 생성
prediction_df = pd.DataFrame()

# 각 건물에 대해 개별적으로 모델을 학습하고 예측
unique_buildings = train_df['building_number'].unique()

for building_num in unique_buildings:
    # 해당 건물에 해당하는 데이터 추출
    building_train_df = train_df[train_df['building_number'] == building_num].copy()

    # Prophet 모델 초기화
    model = Prophet(changepoint_prior_scale=0.1, seasonality_mode='multiplicative', seasonality_prior_scale=10.0)
    # Assuming train_df is already prepared with necessary features and 'power_consumption' column
    building_train_data = building_train_df[['date_time', 'power_consumption']]
    building_train_data = building_train_data.rename(columns={'date_time': 'ds', 'power_consumption': 'y'})

    # 모델 학습
    model.fit(building_train_data)

    # 해당 건물에 대한 예측 수행
    building_test_df = test_df[test_df['building_number'] == building_num].copy()
    building_test_df['date_time'] = pd.to_datetime(building_test_df['date_time'], format='%Y%m%d %H')

    # Test 데이터에 대한 날짜(ds) 컬럼 추가
    building_test_data = building_test_df[['date_time']].rename(columns={'date_time': 'ds'})

    # Test 데이터에 전력소비량(kWh) 컬럼 추가하고 NaN 값으로 채우기
    building_test_data['power_consumption'] = np.nan

    # Test 데이터에 대한 전력소비량 예측
    forecast = model.predict(building_test_data)

    # 예측 결과를 prediction_df에 추가
    building_test_df['power_consumption'] = forecast['yhat'].values
    prediction_df = pd.concat([prediction_df, building_test_df[['date_time', 'power_consumption']]])

# Submission 파일 생성

# test.csv 파일 다시 불러오기
test_df = pd.read_csv('./test.csv')

# 'num_date_time' 컬럼을 prediction_df 데이터프레임에 추가
submission = pd.concat([test_df[['num_date_time']], prediction_df[['power_consumption']]], axis=1)
submission = submission.rename(columns={'power_consumption' : 'answer'})
submission.to_csv('./prophet_submission.csv', index=False)