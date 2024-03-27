import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 데이터 불러오기 및 전처리
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

# Time Series Cross-Validation을 활용하여 모델 평가
tscv = TimeSeriesSplit(n_splits=5)  # 5개 폴드로 나누어 교차 검증
rmse_scores = []  # RMSE 값을 저장할 리스트

for train_index, test_index in tscv.split(train_df):
    # Train 데이터셋과 Test 데이터셋 분리
    train_data = train_df.iloc[train_index]
    test_data = train_df.iloc[test_index]

    # Prophet 모델 초기화
    model = Prophet(changepoint_prior_scale=0.1, seasonality_mode='multiplicative', seasonality_prior_scale=10.0)
    model.add_seasonality(name='season', period=365/4, fourier_order=5)  # 연간 계절성을 추가

    # 모델 학습
    model.fit(train_data[['date_time', 'power_consumption']].rename(columns={'date_time': 'ds', 'power_consumption': 'y'}))

    # Test 데이터에 대한 전력소비량 예측
    forecast = model.predict(test_data[['date_time']].rename(columns={'date_time': 'ds'}))

    # 예측 결과를 prediction_df에 추가
    test_data['power_consumption_pred'] = forecast['yhat'].values
    prediction_df = pd.concat([prediction_df, test_data[['date_time', 'power_consumption_pred']]])

    # RMSE 값 계산하여 저장
    rmse = mean_squared_error(test_data['power_consumption'], test_data['power_consumption_pred'], squared=False)
    rmse_scores.append(rmse)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
sns.lineplot(x='date_time', y='power_consumption', data=train_df, label='Actual')
sns.lineplot(x='date_time', y='power_consumption_pred', data=prediction_df, label='Predicted')
plt.xlabel('Date Time')
plt.ylabel('Power Consumption')
plt.title('Power Consumption Prediction')
plt.legend()
plt.show()

# RMSE 평균 출력
print('Average RMSE:', np.mean(rmse_scores))

# Submission 파일 생성
# test.csv 파일 다시 불러오기
test_df = pd.read_csv('./test.csv')

# 예측 결과를 test_df와 합치기 위해 'num_date_time' 컬럼 다시 생성
test_df['date_time'] = pd.to_datetime(test_df['일시'], format='%Y%m%d %H')
submission = pd.DataFrame()

# Test 데이터셋로 최종 예측 수행
for building_num in test_df['건물번호'].unique():
    # 해당 건물에 해당하는 Test 데이터 추출
    building_test_df = test_df[test_df['건물번호'] == building_num].copy()
    building_test_data = building_test_df[['date_time']].rename(columns={'date_time': 'ds'})

    # Test 데이터에 대한 전력소비량 예측
    forecast = model.predict(building_test_data)

    # 예측 결과를 submission에 추가
    building_test_df['전력소비량(kWh)'] = forecast['yhat'].values
    submission = pd.concat([submission, building_test_df[['num_date_time', '전력소비량(kWh)']]])


submission = submission.rename(columns={'전력소비량(kWh)' : 'answer'})
# submission 파일로 저장
submission.to_csv('./prophet_submission.csv', index=False)