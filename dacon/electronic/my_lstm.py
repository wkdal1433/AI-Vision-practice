import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 로드
train_df = pd.read_csv("./train.csv", encoding='utf-8')
building_info_df = pd.read_csv("./building_info.csv", encoding='utf-8')
test_df = pd.read_csv("./test.csv", encoding='utf-8')

# LSTM 모델에 사용하기 위해 데이터 전처리
def prepare_data(df, building_info_df, is_train=True):
    # 건물 정보를 병합
    df = df.merge(building_info_df, on='건물번호', how='left')

    # 필요한 컬럼만 선택
    selected_cols = ['일시', '건물번호', '전력소비량(kWh)', '기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)']
    df = df[selected_cols]

    # 일시(datetime)를 날짜와 시간으로 분리
    df['일시'] = pd.to_datetime(df['일시'])
    df['날짜'] = df['일시'].dt.date
    df['시간'] = df['일시'].dt.hour

    # 결측치 처리 (0으로 대체)
    df.fillna(0, inplace=True)

    # train 데이터인 경우 전력 사용량에 로그 변환 적용 (대부분의 경우 시계열 데이터는 로그 변환을 통해 정규분포에 가깝게 만들 수 있음)
    if is_train:
        df['전력소비량(kWh)'] = np.log1p(df['전력소비량(kWh)'])


    # 필요 없는 컬럼 제거
    df.drop(columns=['일시', '일조(hr)', '일사(MJ/m2)'], inplace=True)

    return df

# train 데이터 전처리
train_data = prepare_data(train_df, building_info_df)

# test 데이터 전처리
test_data = prepare_data(test_df, building_info_df, is_train=False)

# 건물별로 데이터를 분할하여 LSTM 모델 학습
def train_models(train_data, test_data, building_info_df):
    model_dict = {}
    
    for building_number in building_info_df['건물번호']:
        print(f"Training model for building: {building_number}")
        # 건물별로 데이터 분할
        train_building = train_data[train_data['건물번호'] == building_number]
        test_building = test_data[test_data['건물번호'] == building_number]

        # 건물별로 전력 사용량을 예측하기 위해 날짜와 시간 컬럼 제거
        train_building.drop(columns=['날짜', '시간'], inplace=True)
        test_building.drop(columns=['날짜', '시간'], inplace=True)

        # Features와 Target 데이터 분리
        X_train = train_building.drop(columns=['전력사용량(kWh)'])
        y_train = train_building['전력사용량(kWh)']
        X_test = test_building.drop(columns=['전력사용량(kWh)'])

        # MinMaxScaler를 사용하여 데이터 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # LSTM 모델 구축
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # 모델 학습
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)

        # 테스트 데이터에 대한 예측
        test_building['전력사용량(kWh)'] = model.predict(X_test_scaled)

        # 로그 역변환
        test_building['전력사용량(kWh)'] = np.expm1(test_building['전력사용량(kWh)'])

        # 건물별 예측 결과를 모델 딕셔너리에 저장
        model_dict[building_number] = test_building[['전력사용량(kWh)']]

    return model_dict

# 모델 학습 및 예측
model_dict = train_models(train_data, test_data, building_info_df)

# 결과 저장
submission_df = pd.DataFrame()
submission_df['num_date_time'] = test_df['건물번호'] + '_' + test_df['일시']
submission_df['answer'] = 0  # 초기값 설정

# 모델에서 얻은 예측값을 제출용 데이터프레임에 저장
for building_number, pred_df in model_dict.items():
    submission_df.loc[submission_df['num_date_time'].isin(pred_df.index), 'answer'] = pred_df['전력사용량(kWh)'].values

# 결과 저장
submission_df.to_csv('submission.csv', index=False) 