from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
import pandas as pd
import numpy as np
import random
import os

from tqdm import tqdm

# train.csv 파일에서 데이터를 불러옵니다.
train = pd.read_csv('./train.csv')


# 모든 건물의 예측값을 저장할 빈 리스트를 생성합니다.
all_preds = []

# train 데이터에 있는 모든 건물 번호를 추출합니다.
unique_buildings = train['건물번호'].unique()
iter_num = 1

# 각 건물 번호별로 모델 학습 및 예측을 수행합니다.
for building_num in tqdm(unique_buildings):
    # 건물 번호에 해당하는 학습 데이터를 생성합니다.
    train_data = train[train['건물번호'] == building_num][['일시', '전력소비량(kWh)']]
    train_data['ds'] = pd.to_datetime(train_data['일시'], format='%Y%m%d')
    train_data.set_index('일시', inplace=True)
    train_data.rename(columns={"전력소비량(kWh)": "y"}, inplace=True)
    train_data.dropna(subset=['일시', '전력사용량(kWh)'], inplace=True)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=10,
        n_lags=3,
        n_forecasts=15
    )
    metrics = m.fit(train_data, freq="D")

    # 예측에 사용할 미래 데이터프레임을 생성합니다.
    df_future = m.make_future_dataframe(train_data, periods=15, n_historic_predictions=10)
    forecast = m.predict(df_future)

    # 해당 건물의 마지막 15개 예측값('yhat15')을 가져옵니다.
    preds = forecast['yhat15'][-15:]

    # 최종 수익률(final_return)을 계산합니다.
    final_return = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]
 
    print('{0} 번째 실행중 '.format(iter_num))
    iter_num += 1

    # 해당 건물의 예측값을 all_preds 리스트에 추가합니다.
    all_preds.append(preds)

# 모든 건물의 예측값을 하나의 데이터프레임으로 합칩니다.
preds_df = pd.concat(all_preds, ignore_index=True)

# sample_submission.csv 파일을 불러와서 제출 양식을 가져옵니다.
submission = pd.read_csv('./sample_submission.csv')

# 'num_date_time' 컬럼을 submission 데이터프레임에 생성합니다.
submission['num_date_time'] = submission['num'].astype(str) + '_' + submission['date_time']

# submission 데이터프레임에 해당하는 'num_date_time'에 예측값을 매핑합니다.
submission['answer'] = preds_df['yhat15']

# 최종 결과를 baseline_submission.csv 파일로 저장합니다.
submission.to_csv('./baseline_submission.csv', index=False)