import pandas as pd
import numpy as np
import xgboost as xgb

from lightgbm import LGBMRegressor

from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# 데이터 로드
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
sample_submission = pd.read_csv('./sample_submission.csv')
building_df = pd.read_csv('./building_info.csv')

train_df = train_df.rename(columns={
    '건물번호': 'num',
    '일시': 'date_time',
    '기온(C)': 'temp',
    '강수량(mm)': 'prec',
    '풍속(m/s)': 'wind',
    '습도(%)': 'hum',
    '일조(hr)': 'sun',
    '일사(MJ/m2)': 'solar_radiation',
    '전력소비량(kWh)': 'power'
})
test_df = test_df.rename(columns={
    '건물번호': 'num',
    '일시': 'date_time',
    '기온(C)': 'temp',
    '강수량(mm)': 'prec',
    '풍속(m/s)': 'wind',
    '습도(%)': 'hum',
    '일조(hr)': 'sun',
    '일사(MJ/m2)': 'solar_radiation',
    '전력소비량(kWh)': 'power_consumption'
})


# classify_weekdays 함수 정의
def classify_weekdays(start_date, end_date):
    date_format = "%Y-%m-%d"
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    classification = []
    current_date = start
    
    while current_date <= end:
        if current_date.weekday() < 5:  # 0: 월요일, 1: 화요일, ..., 4: 금요일
            classification.append(0)  # 평일
        else:
            classification.append(1)  # 주말
        
        current_date += timedelta(days=1)
    
    return classification

# 예시 데이터 프레임 생성
data = {'date': pd.date_range(start='2022-06-01', end='2022-08-24')}
train = pd.DataFrame(data)
t_data = {'date': pd.date_range(start='2022-08-25', end='2022-08-31')}
test = pd.DataFrame(t_data)

# 주말/평일 분류한 결과를 'holidays' 열에 추가
start_date = "2022-06-01"
end_date = "2022-08-24"
test_start_date = "2022-08-25"
test_end_date = "2022-08-31"
train['holidays'] = classify_weekdays(start_date, end_date)
train_df['holidays'] = train['holidays']
test['holidays'] = classify_weekdays(test_start_date, test_end_date)
test_df['holidays'] = test['holidays']

# '일시' 열을 datetime 형식으로 변환
train_df['date_time'] = pd.to_datetime(train_df['date_time'], format='%Y%m%d %H')
test_df['date_time'] = pd.to_datetime(test_df['date_time'], format='%Y%m%d %H')

# date time feature 생성
train_df['hour'] = train_df['date_time'].dt.hour
train_df['day'] = train_df['date_time'].dt.day
train_df['month'] = train_df['date_time'].dt.month
train_df['year'] = train_df['date_time'].dt.year
train_df['sin_time'] = np.sin(2*np.pi*train_df.hour/24)
train_df['cos_time'] = np.cos(2*np.pi*train_df.hour/24)
train_df['THI'] = 9/5*train_df['temp'] - 0.55*(1-train_df['hum']/100)*(9/5*train_df['hum']-26)+32

test_df['hour'] = test_df['date_time'].dt.hour
test_df['day'] = test_df['date_time'].dt.day
test_df['month'] = test_df['date_time'].dt.month
test_df['year'] = test_df['date_time'].dt.year
test_df['sin_time'] = np.sin(2*np.pi*test_df.hour/24)
test_df['cos_time'] = np.cos(2*np.pi*test_df.hour/24)
test_df['THI'] = 9/5*test_df['temp'] - 0.55*(1-test_df['hum']/100)*(9/5*test_df['hum']-26)+32

# 필요한 열 삭제 (num_date_time 포함)
train_df = train_df.drop(columns=['sun', 'solar_radiation', 'prec', 'wind', 'num_date_time'])
test_df = test_df.drop(columns=['prec', 'wind', 'num_date_time'])

# 예측 모델 학습 데이터 생성
train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)
# datetime 열을 숫자로 변환
train_data['date_time'] = (train_data['date_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
test_data['date_time'] = (test_data['date_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# 레이블 정의
label = 'power'

# 첫 번째 단계 모델 생성: Random Forest와 XGBoost
model_rf = LGBMRegressor()
model_xgb = xgb.XGBRegressor()

# 첫 번째 단계 모델 학습
model_rf.fit(train_data.drop(columns=[label]), train_data[label])
model_xgb.fit(train_data.drop(columns=[label]), train_data[label])

# 첫 번째 단계 모델 예측
pred_rf = model_rf.predict(test_data)
pred_xgb = model_xgb.predict(test_data)

# 두 번째 단계 모델 입력 데이터 생성
test_data['pred_rf'] = pred_rf
test_data['pred_xgb'] = pred_xgb

# 두 번째 단계 모델 학습: AutoGluon을 사용하여 스태킹 앙상블 모델 학습
stacking_predictor = TabularPredictor(label=label, problem_type='regression').fit(
    train_data, presets=['best_quality'], auto_stack=True, num_stack_levels=1
)

# 최종 예측
stacking_pred = stacking_predictor.predict(test_data)
final_pred = stacking_pred[label]

# 결과 출력
sample_submission['answer'] = final_pred
print(sample_submission)
sample_submission.to_csv('./stacked_submission.csv', index=False)