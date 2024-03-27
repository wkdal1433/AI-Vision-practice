from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
import pandas as pd
import numpy as np
import random
import os

from tqdm import tqdm


train = pd.read_csv('./train.csv')

# 추론 결과를 저장하기 위한 dataframe 생성
results_df = pd.DataFrame(columns=['종목코드', 'final_return'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()
iter_num = 1
# 각 종목코드에 대해서 모델 학습 및 추론 반복
for code in tqdm(unique_codes):
    # 학습 데이터 생성
    train_close = train[train['종목코드'] == code][['일자', '종가']]
    train_close['ds'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)
    # df_new = df.drop(columns=['year','month','day'])
    train_close.rename(columns={"종가": "y"},inplace=True)
    # print(train_close)
    m = NeuralProphet(
        # growth='off', # 추세 유형 설정(linear, discontinuous, off 중 선택 가능)
        yearly_seasonality=False, #년간 계절성 설정

        weekly_seasonality=False, #주간 계절성 설정

        daily_seasonality=False, #일간 계절성 설정


        epochs=10,#학습 횟수 설정

        # learning_rate=0.01, # 학습률 설정

        n_lags = 3,
        n_forecasts = 15

    )
    metrics = m.fit(train_close, freq="D")

    df_future = m.make_future_dataframe(train_close, periods=15,  n_historic_predictions=10)
    forecast = m.predict(df_future)

    preds = forecast['yhat15'][-15:]

    final_return = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]
    # 결과 저장
    results_df = results_df.append({'종목코드': code, 'final_return': final_return}, ignore_index=True)
    print('{0} 번째 실행중 '.format(iter_num))
    iter_num+=1


results_df['순위'] = results_df['final_return'].rank(method='first', ascending=False).astype('int') # 각 순위를 중복없이 생성

sample_submission = pd.read_csv('./sample_submission.csv')

baseline_submission = sample_submission[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')

baseline_submission.to_csv('baseline_submission.csv', index=False)