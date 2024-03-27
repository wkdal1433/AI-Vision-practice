import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# 데이터 로드
train_df = pd.read_csv("./train.csv")
building_info_df = pd.read_csv("./building_info.csv")
test_df = pd.read_csv("./test.csv")

# train 데이터 전처리
def prepare_data(df, building_info_df, is_train=True):
    # 건물 정보를 병합
    df = df.merge(building_info_df, on='건물번호', how='left')

    # 필요한 컬럼만 선택
    selected_cols = ['건물번호', '전력소비량(kWh)', '기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    df = df[selected_cols]

    # 결측치 처리 (0으로 대체)
    df.fillna(0, inplace=True)

    return df

# train 데이터 전처리
train_data = prepare_data(train_df, building_info_df)

# Features와 Target 데이터 분리
X = train_data.drop(columns=['전력소비량(kWh)'])
y = train_data['전력소비량(kWh)']

# MinMaxScaler를 사용하여 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Random Forest 모델 학습
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_scaled, y)

# test 데이터 전처리
test_data = prepare_data(test_df, building_info_df, is_train=False)

# MinMaxScaler를 사용하여 데이터 스케일링
test_data_scaled = scaler.transform(test_data.drop(columns=['건물번호']))

# 건물별로 test 데이터 예측
preds = rf_model.predict(test_data_scaled)

# 결과 파일 생성
submission_df = pd.DataFrame()
submission_df['num_date_time'] = test_df['건물번호'] + '_' + test_df['일시']
submission_df['answer'] = preds

# 결과 저장
submission_df.to_csv('submission.csv', index=False)