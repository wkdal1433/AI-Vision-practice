import pandas as pd
from prophet import Prophet

# train 데이터 불러오기
train_df = pd.read_csv('train.csv')

# building_info 데이터 불러오기
building_info_df = pd.read_csv('building_info.csv')

# test 데이터 불러오기
test_df = pd.read_csv('test.csv')

# sample_submission 데이터 불러오기
sample_submission_df = pd.read_csv('sample_submission.csv')

# train 데이터 전처리
train_df['ds'] = pd.to_datetime(train_df['일시'], format='%Y%m%d %H')
train_df['y'] = train_df['전력소비량(kWh)']

# building_info 데이터 전처리
building_info_df.rename(columns={'건물번호': '건물번호', '태양광용량(kW)': '태양광용량'}, inplace=True)

# test 데이터 전처리
test_df['ds'] = pd.to_datetime(test_df['일시'], format='%Y%m%d %H')

# 추가적인 특성 생성 (기온, 강수량, 풍속, 습도, 일조, 일사 정보 활용 가능)
train_df['temperature'] = train_df['기온(C)']
train_df['rainfall'] = train_df['강수량(mm)']
train_df['windspeed'] = train_df['풍속(m/s)']
train_df['humidity'] = train_df['습도(%)']

test_df['temperature'] = test_df['기온(C)']
test_df['rainfall'] = test_df['강수량(mm)']
test_df['windspeed'] = test_df['풍속(m/s)']
test_df['humidity'] = test_df['습도(%)']

# 결측치 대체
train_df['rainfall'].fillna(train_df['rainfall'].mean(), inplace=True)
test_df['rainfall'].fillna(test_df['rainfall'].mean(), inplace=True)

train_df['windspeed'].fillna(train_df['windspeed'].mean(), inplace=True)
test_df['windspeed'].fillna(test_df['windspeed'].mean(), inplace=True)

train_df['humidity'].fillna(train_df['humidity'].mean(), inplace=True)
test_df['humidity'].fillna(test_df['humidity'].mean(), inplace=True)


# 건물별로 모델을 학습하고 예측하여 결과를 저장할 DataFrame 초기화
submission_df = pd.DataFrame()

# 건물별로 반복하여 예측 수행
for building_num in train_df['건물번호'].unique():
    # 건물별로 train 데이터 준비
    building_train_df = train_df[train_df['건물번호'] == building_num]

    # Prophet 모델 초기화
    model = Prophet()

    # 특성 추가
    model.add_regressor('temperature')
    model.add_regressor('rainfall')
    model.add_regressor('windspeed')
    model.add_regressor('humidity')

    # 모델 학습
    model.fit(building_train_df[['ds', 'y', 'temperature', 'rainfall', 'windspeed', 'humidity']])

    # test 데이터 준비
    building_test_df = test_df[test_df['건물번호'] == building_num]

    # 예측
    future = building_test_df[['ds', 'temperature', 'rainfall', 'windspeed', 'humidity']]
    forecast = model.predict(future)

    # 결과 저장
    building_test_df['전력사용량(kWh)_예측'] = forecast['yhat'].values
    submission_df = pd.concat([submission_df, building_test_df[['num_date_time', '전력사용량(kWh)_예측']]])

# sample_submission과 합치기
submission_df = submission_df.merge(sample_submission_df, on='num_date_time')

# 결과 저장
submission_df.drop(columns=['전력사용량(kWh)_예측_x'], inplace=True)
submission_df.rename(columns={'전력사용량(kWh)_예측_y': 'answer'}, inplace=True)
submission_df.to_csv('submission.csv', index=False)
