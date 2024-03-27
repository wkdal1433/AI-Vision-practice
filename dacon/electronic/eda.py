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
train_df.drop('num_date_time', axis = 1, inplace=True)

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
test_df.drop('num_date_time', axis = 1, inplace=True)


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

len(train_df)
train_df.isna().sum()
test_df.isna().sum()

print(len(train_df[train_df['solar_power_capacity'] == '-']))
print(len(train_df[train_df['ess_capacity'] == '-']))
print(len(train_df[train_df['pcs_capacity'] == '-']))

train_df = train_df.drop(['solar_power_capacity', 'ess_capacity', 'pcs_capacity'], axis=1)
test_df = test_df.drop(['solar_power_capacity', 'ess_capacity', 'pcs_capacity'], axis=1)

train_df['date_time'] = pd.to_datetime(train_df['date_time'], format='%Y%m%d %H')

# date time feature 생성
train_df['hour'] = train_df['date_time'].dt.hour
train_df['day'] = train_df['date_time'].dt.day
train_df['month'] = train_df['date_time'].dt.month
train_df['year'] = train_df['date_time'].dt.year


# Prophet 모델 초기화
model = Prophet(changepoint_prior_scale=0.05)

# Assuming train_df is already prepared with necessary features and 'power_consumption' column
train_data = train_df[['date_time', 'power_consumption']]
train_data = train_data.rename(columns={'date_time': 'ds', 'power_consumption': 'y'})

# 모델 학습
model.fit(train_data)

# Test 데이터에 대한 날짜(ds) 컬럼 추가
test_data = test_df[['date_time']].rename(columns={'date_time': 'ds'})

# Test 데이터에 전력소비량(kWh) 컬럼 추가하고 NaN 값으로 채우기
test_data['power_consumption'] = np.nan

# Test 데이터에 대한 전력소비량 예측
forecast = model.predict(test_data)

# 예측 결과를 test_df에 추가
test_df['power_consumption'] = forecast['yhat'].values

# Submission 파일 생성
submission = test_df[['power_consumption']]
# test.csv 파일 다시 불러오기
test_df = pd.read_csv('./test.csv')

# 'num_date_time' 컬럼을 submission 데이터프레임의 첫 번째 컬럼으로 추가
submission.insert(0, 'num_date_time', test_df['num_date_time'])

submission.to_csv('./prophet_submission.csv', index=False)

